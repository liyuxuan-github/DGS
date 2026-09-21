import torch
from scipy.stats import norm,truncnorm
from functools import reduce
from scipy.special import betainc
import numpy as np
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
from tqdm import  tqdm


import random
from itertools import combinations


class TensorGenerator:
    def __init__(self, batch_size=1, channels=4, height=64, width=56, num_masked=1024):
        """Initializes the class with tensor shape and the number of masked positions."""
        self.batch_size = batch_size
        self.channels = channels
        self.height = height  
        self.width = width  
        self.num_masked = num_masked  # Number of positions to mask per selection
        
        # Generate 16 sets of masked positions for each channel
        self.masked_positions_sets = self.generate_masked_positions()

        # Initialize with random tensors
        self.reset(torch.randint(0, 2, (self.batch_size, self.channels, self.height, self.width)),
                   torch.randint(0, 2, (self.batch_size, self.channels, 4, 1)))  # 4-bit keys (1,4,4,1)

    def reset(self, B, key):
        """Resets the tensor and selects the corresponding masked positions."""
        self.B = B
        self.binary_tensor = key
        self.decimal_indices = self.convert_binary_to_index(self.binary_tensor)
        self.selected_masked_positions = self.get_selected_masked_positions()
        self.calculate_float_tensor()

    # ==============================================
    # 1. Generate 16 sets of random masked positions for each channel
    # ==============================================
    def generate_masked_positions(self):
        """Generates 16 different sets of random masked positions per channel."""
        masked_positions_per_channel = {}

        for c in range(self.channels):  # Iterate over channels
            masked_positions_per_channel[c] = [
                random.sample([(h, w) for h in range(self.height) for w in range(self.width)], self.num_masked)
                for _ in range(16)  # Generate 16 different selections
            ]

        return masked_positions_per_channel

    # ==============================================
    # 2. Convert binary tensor (1,4,4,1) to decimal indices (0-15)
    # ==============================================
    def convert_binary_to_index(self, binary_tensor):
        """Convert 4-bit binary tensor into decimal indices (0-15)."""
        binary_tensor_flat = binary_tensor.squeeze(-1)  # Shape: (1,4,4)
        return (binary_tensor_flat * torch.tensor([8, 4, 2, 1])).sum(dim=-1)  # Shape: (1,4)

    # ==============================================
    # 3. Get masked positions based on the selected key index
    # ==============================================
    def get_selected_masked_positions(self):
        """Selects the masked positions based on the decimal index for each channel."""
        selected_masked_positions = {}

        for c in range(self.channels):  # Iterate over channels
            selected_index = self.decimal_indices[0, c]  # Get the selected index (0-15) from the key
            selected_masked_positions[c] = self.masked_positions_sets[c][selected_index]  # Get the corresponding masked positions

        return selected_masked_positions

    # ==============================================
    # 4. Generate Gaussian samples for filling the tensor
    # ==============================================
    def generate_gaussian_samples(self, p, n):
        """Generate Gaussian samples and split into positive, negative, and remainder sets."""
        while True:
            samples = np.random.randn(14336)  # Generate samples from a standard Gaussian distribution
            positive_samples = samples[samples > 0]
            negative_samples = samples[samples < 0]

            if len(positive_samples) > p and len(negative_samples) > n:
                break

        positive_part = np.partition(positive_samples, -p)[-p:]  # Largest `p` positive values
        negative_part = np.partition(negative_samples, n)[:n]  # Smallest `n` negative values
        remaining_samples = np.setdiff1d(samples, np.concatenate([positive_part, negative_part]))

        return positive_part, negative_part, remaining_samples

    # ==============================================
    # 5. Calculate the float tensor values
    # ==============================================
    def calculate_float_tensor(self):
        """Computes the float tensor using the binary tensor and masked positions."""
        positive_count = 0
        negative_count = 0

        for b in range(self.batch_size):
            for c in range(self.channels):
                for h in range(self.height):
                    for w in range(self.width):
                        if (h, w) in self.selected_masked_positions[c]:  # If this position is masked
                            continue
                        if self.B[b, c, h, w] == 1:
                            positive_count += 1
                        else:
                            negative_count += 1

        # Generate Gaussian-distributed values for each category
        self.positive_set, self.negative_set, self.remainder_set = self.generate_gaussian_samples(positive_count, negative_count)

    # ==============================================
    # 6. Generate the final float tensor
    # ==============================================
    def generate_float_tensor(self):
        """Generates (1,4,64,64) float tensor using B and masked position information."""
        output_tensor = torch.zeros(self.batch_size, self.channels, self.height, self.width)

        positive_list = list(self.positive_set)
        negative_list = list(self.negative_set)
        remainder_list = list(self.remainder_set)
        
        random.shuffle(positive_list)
        random.shuffle(negative_list)
        random.shuffle(remainder_list)

        for b in range(self.batch_size):  # Batch loop
            for c in range(self.channels):  # Channel loop
                for h in range(self.height):  # Row loop
                    for w in range(self.width):  # Column loop
                        if (h, w) in self.selected_masked_positions[c]:  # Masked positions → remainder set
                            value = remainder_list.pop()
                        else:
                            if self.B[b, c, h, w] == 1:  # B == 1 → positive set
                                value = positive_list.pop()
                            else:  # B == 0 → negative set
                                value = negative_list.pop()

                        output_tensor[b, c, h, w] = value  # Assign sampled value

        return output_tensor

    # ==============================================
    # 7. Function to mask a given tensor using the selected positions
    # ==============================================
    def mask_tensor(self, w , key):
        """Sets masked positions to 0 in the given tensor w."""
        key = key.unsqueeze(0)
        self.binary_tensor = key
        self.decimal_indices = self.convert_binary_to_index(self.binary_tensor)
        self.selected_masked_positions = self.get_selected_masked_positions()
        for b in range(self.batch_size):  # Batch loop
            for c in range(self.channels):  # Channel loop
                for h, w_pos in self.selected_masked_positions[c]:  # Iterate over masked positions
                    w[b, c, h, w_pos] = 0.0
        return w

class Gaussian_Shading_chacha:
    def __init__(self, ch_factor, hw_factor, fpr, user_number , keepuser=False):
        self.ch = ch_factor
        self.hw = hw_factor
        self.nonce = None
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None
        self.keepuser=keepuser

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

        self.key = get_random_bytes(32)
        self.nonce = get_random_bytes(12)
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()

    def stream_key_encrypt(self, sd):
        if not self.keepuser:
            self.key = get_random_bytes(32)
            self.nonce = get_random_bytes(12)
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z.cuda()

    def create_watermark_and_return_w(self):
        if not self.keepuser:
            self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()
        sd = self.watermark.repeat(1,self.ch,self.hw,self.hw)
        m = self.stream_key_encrypt(sd.flatten().cpu().numpy())
        w = self.truncSampling(m)
        return w

    def stream_key_decrypt(self, reversed_m):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, 64).to(torch.uint8)
        return sd_tensor.cuda()

    def diffusion_inverse(self,watermark_r):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_w):
        reversed_m = (reversed_w > 0).int()
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        print(correct)
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count




class Dynamic_Gaussian_Shading_chacha:
    def __init__(self, ch_factor, hw_factor, fpr, user_number , num_masked=1024,keepuser=False):
        self.ch = ch_factor
        self.hw = hw_factor
        self.nonce = None
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None
        self.keepuser=keepuser

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

        self.key = get_random_bytes(32)
        self.nonce = get_random_bytes(12)
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()
        self.tg=TensorGenerator(num_masked=num_masked)

    def stream_key_encrypt(self, sd):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit

    def truncSampling(self, message):
        length=len(message)
        z = np.zeros(length)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(length):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, -1).half()
        return z.cuda()

    def create_watermark_and_return_w(self):
        if not self.keepuser:
            self.key = get_random_bytes(32)
            self.nonce = get_random_bytes(12)
        if not self.keepuser:
            self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()
        
        random_mark= torch.randint(0, 2, [1, 4 // self.ch, 32 // self.hw, 1]).cuda()
        self.m=random_mark
        sd = random_mark.repeat(1,self.ch,self.hw*2,self.hw)
        random_m = self.stream_key_encrypt(sd.flatten().cpu().numpy())
        
        merged_watermark=(self.watermark==random_mark.repeat(1,1,2,8)).int()
        sd = merged_watermark.repeat(1,self.ch,self.hw,self.hw-1)
        m = self.stream_key_encrypt(sd.flatten().cpu().numpy())
        
        
        
        w1 = self.truncSampling(random_m)
        #w2 = self.truncSampling(m)
        w2 = torch.from_numpy(m).reshape(1, 4, 64, -1).int()
        self.tg.reset(w2.cpu(),random_mark.cpu())
        w2=self.tg.generate_float_tensor().cuda().half()
        w0,w2=torch.split(w2,(24,32),dim=3)
        w=torch.cat((w0,w1,w2),dim=-1)
    
        self.w=(w>0).int().flatten()
        return w

    def stream_key_decrypt(self, reversed_m):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, -1).to(torch.uint8)
        return sd_tensor.cuda()

    def diffusion_inverse(self,watermark_r):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        h_list = [hw_stride] * self.hw
        w_list = [hw_stride] * (self.hw//2)
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(h_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(w_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold//2] = 0
        vote[vote > self.threshold//2] = 1
        return vote
    
    def repeat_inverse(self,watermark_r,mode=False):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        if mode:
            h_list = [hw_stride//2] * (self.hw*2) 
            w_list = [1] * (self.hw * 1)
        else:
            h_list = [hw_stride] * self.hw
            w_list = [hw_stride] * (self.hw-1)
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(h_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(w_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        
        return (vote>0).int()

    def eval_watermark(self, reversed_w):
        w0,w1,w2=torch.split(reversed_w,(24,8,32),dim=3)
        w2=torch.cat((w0,w2),dim=3)
        reversed_m1 = (w1 > 0).int()
        weight_mask1=torch.abs(w1)
        reversed_sd1 = self.stream_key_decrypt(reversed_m1.flatten().cpu().numpy())
        #reversed_watermark = self.diffusion_inverse(reversed_sd)
        reversed_m1 = self.repeat_inverse((reversed_sd1.float()-0.5)*weight_mask1,True)
        rm=reversed_m1
        
        
        reversed_m2 = (w2 > 0).int()
        weight_mask2=torch.abs(w2)
        reversed_sd2 = self.stream_key_decrypt(reversed_m2.flatten().cpu().numpy())
        reversed_sd2 = ((reversed_sd2).float()-0.5)*weight_mask2
        reversed_sd2 = self.tg.mask_tensor(reversed_sd2,rm.cpu())
        reversed_m2 = self.repeat_inverse(reversed_sd2)
        
        reversed_watermark = (reversed_m1.repeat(1,1,2,8)==reversed_m2).int()
        
        
        correct = (reversed_watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
            
        reversed_w = (reversed_w > 0).int().flatten()
        print(correct,torch.sum(reversed_w==self.w),torch.sum(self.m==rm)/len(rm.flatten()))
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count
