# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
from builtins import str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jsons", type=str, nargs="+", help="json files")
    parser.add_argument(
        "--multi",
        "-m",
        type=int,
        help="Test the json file for multiple input/output",
        default=0,
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    args = parser.parse_args()

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )

    # make intersection set for utterance keys
    js = []
    intersec_ks = []
    for x in args.jsons:
        with open(x, "r") as f:
            j = json.load(f)
        ks = j["utts"].keys()
        logging.info(x + ": has " + str(len(ks)) + " utterances")
        if len(intersec_ks) > 0:
            intersec_ks = intersec_ks.intersection(set(ks))
        else:
            intersec_ks = set(ks)
        js.append(j)
    logging.info("new json has " + str(len(intersec_ks)) + " utterances")

    old_dic = dict()
    for k in intersec_ks:
        v = js[0]["utts"][k]
        for j in js[1:]:
            v.update(j["utts"][k])
        old_dic[k] = v

    new_dic = {}
    for i in old_dic:
        dic = old_dic[i]

        in_dic = {}
        if dic.has_key(str("idim")):
            in_dic[str("shape")] = (
                int(dic[str("ilen")]),
                int(dic[str("idim")]),
            )
        in_dic[str("name")] = str("input1")
        in_dic[str("feat")] = dic[str("feat")]

        out_dic = {}
        out_dic[str("name")] = str("target1")
        out_dic[str("shape")] = (
            int(dic[str("olen")]),
            int(dic[str("odim")]),
        )
        out_dic[str("text")] = dic[str("text")]
        out_dic[str("token")] = dic[str("token")]
        out_dic[str("tokenid")] = dic[str("tokenid")]

        new_dic[id] = {
            str("input"): [in_dic],
            str("output"): [out_dic],
            str("utt2spk"): dic[str("utt2spk")],
        }

    # ensure "ensure_ascii=False", which is a bug
    jsonstring = json.dumps(
        {"utts": new_dic}, indent=4, ensure_ascii=False, sort_keys=True
    ).encode("utf_8")
    print(jsonstring)
