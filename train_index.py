import argparse
import os
import pickle

import utils

import configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", type=str, default="", help="speaker name")
    parser.add_argument("--source_dir", type=str, default=configs.data_dir, help="path to source dir")
    parser.add_argument("--config_dir", type=str, default=configs.data_dir, help="path to config dir")
    parser.add_argument("--model_dir", type=str, default=configs.model_dir, help="path to val list")

    args = parser.parse_args()

    if args.speaker == "":
        raise Exception("type speaker")

    config_path = os.path.join(args.config_dir, args.speaker, "configs", "config.json")

    hps = utils.get_hparams_from_file(config_path)
    spk_dic = hps.spk
    result = {}
    
    for k,v in spk_dic.items():
        print(f"now, index {k} feature...")
        index = utils.train_index(k,os.path.join(args.source_dir, args.speaker, "44k"))
        result[v] = index

    with open(os.path.join(os.path.join(args.model_dir, args.speaker),"feature_and_index.pkl"),"wb") as f:
        pickle.dump(result,f)