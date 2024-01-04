#!/bin/env bash

for i in {01..20}; do
	mkdir -p sub_${i}
	cp input.toml q_opt makefile sub_${i}/.
	sed -i "/#SBATCH -J/c\#SBATCH -J QmostPAN_${i}" sub_${i}/q_opt
done
