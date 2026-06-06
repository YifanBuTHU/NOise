#!/bin/bash

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2 --no-check-certificate
  rm -rf /tmp/cookies.txt
}

GDRIVE_ID=1kHJUqb-e7BARb63741DVdpg-WqCdG3z6
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_DIR="$PROJECT_ROOT/experiments"
TAR_FILE="$EXP_DIR/pretrained.tar"

mkdir -p "$EXP_DIR"

gdrive_download $GDRIVE_ID "$TAR_FILE"
tar -xvf "$TAR_FILE" -C "$EXP_DIR/"
rm "$TAR_FILE"
