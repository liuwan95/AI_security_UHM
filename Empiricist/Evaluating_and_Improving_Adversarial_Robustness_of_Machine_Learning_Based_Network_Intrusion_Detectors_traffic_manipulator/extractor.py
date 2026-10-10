import argparse
import sys
import AfterImageExtractor.FEKitsune as Fe
from AfterImageExtractor.KitsuneTools import RunFE
import numpy as np
from scapy.all import *

if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument('-i', '--input_path', type=str, required=True, help="raw traffic (.pcap) path")
    parse.add_argument('-o', '--output_path', type=str, required=True, help="feature vectors (.npy) path")

    arg = parse.parse_args()
    pcap_file = arg.input_path

    feat_file = arg.output_path

    scapyin = rdpcap(pcap_file)

    # Was: np.Inf — NumPy 2.x removed Inf alias; use np.inf
    FE = Fe.Kitsune(scapyin, np.inf)
    feature, _ = RunFE(FE)

    print(np.asarray(feature).shape)
    np.save(feat_file,feature)

            
