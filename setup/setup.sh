#!/bin/bash
tmp_tag="tmp-$(cat /dev/urandom | tr -dc 'a-z' | fold -w 32 | head -n 1)"
docker build \
       -f Dockerfile \
       -t "$tmp_tag" \
       .

tmp_tag="utcnpjpdbsorhktnnnttixmkvlyxirwl"

function drun() {
  docker run --rm "$tmp_tag" $*
}
