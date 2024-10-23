#!/usr/bin/env bash
cd $(dirname $0)

nixfmt *.nix

isort -q --profile black --no-sections jax_optics
black -q jax_optics
flake8 --max-line-length 100 jax_optics
