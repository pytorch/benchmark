import argparse
import pathlib
import yaml

CORE_MODEL_PATH = pathlib.Path(__file__).parent.parent.absolute().joinpath("torchbenchmark", "models")

def get_model_list():
    models = list(map(lambda x: x.name, filter(lambda x: x.is_dir(), CORE_MODEL_PATH.iterdir())))
    return models

def check_csv_file(csv_file, known_models):
    optimial_bsizes = {}
    with open(csv_file, "r") as cf:
        opt_csv = cf.readlines()
    for line in opt_csv:
        model, bsize = line.split(",")
        bsize = int(bsize)
        if not bsize == 0:
            optimial_bsizes[model] = bsize
        assert model in known_models, f"Model {model} is not covered in TorchBench core model list."
    return optimial_bsizes

def update_model_optimal_bsizes(device, new_bsize, known_models):
    # get the metadata of exist model
    for model in new_bsize:
        metadata_path = CORE_MODEL_PATH.joinpath(model).joinpath("metadata.yaml")
        with open(metadata_path, "r") as mp:
            metadata = yaml.safe_load(mp)
        if not "devices" in metadata or device not in metadata["devices"]:
            metadata["devices"] = {}
            metadata["devices"][device] = {}
            metadata["devices"][device]["eval_batch_size"] = new_bsize[model]
        with open(metadata_path, "w") as mp:
            yaml.safe_dump(metadata, mp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, type=str, help="Name of the device")
    parser.add_argument("--optimal-bsize-csv", required=True, type=str, help="Optimal Batchsize CSV file")
    args = parser.parse_args()

    known_models = get_model_list()
    new_bsize = check_csv_file(args.optimal_bsize_csv, known_models)
    update_model_optimal_bsizes(args.device, new_bsize, known_models)
