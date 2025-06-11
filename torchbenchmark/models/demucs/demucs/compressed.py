# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

import musdb

from .audio import AudioFile


def get_musdb_tracks(root, *args, **kwargs):
    mus = musdb.DB(root, *args, **kwargs)
    return {track.name: track.path for track in mus}


class StemsSet:
    def __init__(
        self, tracks, metadata, duration=None, stride=1, samplerate=44100, channels=2
    ):
        self.metadata = []
        for name, path in tracks.items():
            meta = dict(metadata[name])
            meta["path"] = path
            meta["name"] = name
            self.metadata.append(meta)
            if duration is not None and meta["duration"] < duration:
                raise ValueError(
                    f"Track {name} duration is too small {meta['duration']}"
                )
        self.metadata.sort(key=lambda x: x["name"])
        self.duration = duration
        self.stride = stride
        self.channels = channels
        self.samplerate = samplerate

    def __len__(self):
        return sum(self._examples_count(m) for m in self.metadata)

    def _examples_count(self, meta):
        if self.duration is None:
            return 1
        else:
            return int((meta["duration"] - self.duration) // self.stride + 1)

    def track_metadata(self, index):
        for meta in self.metadata:
            examples = self._examples_count(meta)
            if index >= examples:
                index -= examples
                continue
            return meta

    def __getitem__(self, index):
        for meta in self.metadata:
            examples = self._examples_count(meta)
            if index >= examples:
                index -= examples
                continue
            streams = AudioFile(meta["path"]).read(
                seek_time=index * self.stride,
                duration=self.duration,
                channels=self.channels,
                samplerate=self.samplerate,
            )
            return (streams - meta["mean"]) / meta["std"]


def _get_track_metadata(path):
    # use mono at 44kHz as reference. For any other settings data won't be perfectly
    # normalized but it should be good enough.
    audio = AudioFile(path)
    mix = audio.read(streams=0, channels=1, samplerate=44100)
    return {
        "duration": audio.duration,
        "std": mix.std().item(),
        "mean": mix.mean().item(),
    }


def build_metadata(tracks):
    return {name: _get_track_metadata(path) for name, path in tracks.items()}


def build_musdb_metadata(path, musdb, workers):
    tracks = get_musdb_tracks(musdb)
    metadata = build_metadata(tracks)
    path.parent.mkdir(exist_ok=True, parents=True)
    json.dump(metadata, open(path, "w"))
