# Merlin's Mirror

## Overview

Merlin's Mirror is a live art installation thatcreates real-time AI videos from a webcam using a Stable Diffusion model.

## Details

Thanks to recent progress in Stable Diffusions models and hardware acceleration, it is now possible to generate images in real-time.

We build on this to create Merlin's Mirror: a live art installation that transforms the world in front of you. It's been hosted at the following places in NYC:
- Merlin's Place
- Telos Haus

The first versions of Merlin's Mirror were not real-time. They ran at about 1 frame per second, with a ~3 second lag from our movements to the live video. This latest version, `mirror_ai`, is fully real-time. 

People have really enjoyed Merlin's Mirror. There are other projects in a similar spirit, but I had the good fortune of hosting this at smaller, fun events where many folks knew each other. This felt quite different from a typical, more impersonal museum or gallery setup. People were collaborating on the prompts, asking their friends to join, and had fun trying to break the model.  

## Accessible Merlin  

This repo is an attempt to make Merlin's Mirror useable by other people. As of writing (04/02/2025), the codebase deploys a Docker container on a machine with your own GPU. Our tests used a graciously borrowed 4090 GPU from Merlin's Place. There is some work to get the code wired to cloud GPUs, but I am still searching for a good provider. If you know of any, please let me know! 

## Tech Stack

- FastHTML and MonsterUI for the frontend
- HuggingFace for the video generation pipeline
- OpenCV for the webcam stream
- Docker for containerization