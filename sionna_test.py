#Imports
import sionna.phy
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
import torch

#Variables
#The seed here basically makes it so you get the same number each time from our random number generator, and depending on the number generation method you use you will get a different number (but the same on each iteration)
#sionna.phy.config.seed = 40

NUM_BITS_PER_SYMBOL = 2
'''
constellation = sionna.phy.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL)
mapper = sionna.phy.mapping.Mapper(constellation=constellation)
demapper = sionna.phy.mapping.Demapper("app", constellation=constellation)

binary_source = sionna.phy.mapping.BinarySource()
awgn_channel = sionna.phy.channel.AWGN()

no = sionna.phy.utils.ebnodb2no(ebno_db=10.0, num_bits_per_symbol=NUM_BITS_PER_SYMBOL, coderate=1.0)

BATCH_SIZE = 64
bits = binary_source([BATCH_SIZE, 1024])
x = mapper(bits)
y = awgn_channel(x, no)
llr = demapper(y, no)

#Code
#print(sionna.phy.config.np_rng.integers(0,10))
constellation.show();

print("Shape of bits: ", bits.shape)
print("Shape of x: ", x.shape)
print("Shape of y: ", y.shape)
print("Shape of llr: ", llr.shape)

plt.figure(figsize = (8, 8))
plt.axes().set_aspect(1)
plt.grid(True)
plt.title('Channel output')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.scatter(y.cpu().real.flatten().numpy(), y.cpu().imag.flatten().numpy())
plt.tight_layout()

plt.show()
'''

class UncodedSystemAWGN(sionna.phy.Block):
    def __init__(self, num_bits_per_symbol, block_length):

        super().__init__() # Must call the block initializer

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = block_length
        self.constellation = sionna.phy.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sionna.phy.mapping.Mapper(constellation=self.constellation)
        self.demapper = sionna.phy.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sionna.phy.mapping.BinarySource()
        self.awgn_channel = sionna.phy.channel.AWGN()

    # @torch.compile # Enable compilation to speed things up
    def call(self, batch_size, ebno_db):

        # no channel coding used; we set coderate=1.0
        no = sionna.phy.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)

        bits = self.binary_source([batch_size, self.block_length]) # Blocklength set to 1024 bits
        x = self.mapper(bits)
        y = self.awgn_channel(x, no)
        llr = self.demapper(y,no)
        return bits, llr

model_uncoded_awgn = UncodedSystemAWGN(num_bits_per_symbol=NUM_BITS_PER_SYMBOL, block_length=1024)


