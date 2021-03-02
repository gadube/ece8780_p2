#!/bin/sh

[ -f '~/hpcgpu/ece8780_p2/hackathon2/build/' ] || mkdir build
rm -rf build/*

for bsize in 12 16 32
do
	sed -i "s/BLOCK=.*/BLOCK=${bsize}/" Makefile
	for sm in 35 60 70
	do
		sed -i "s/sm\_../sm\_$sm/" Makefile
		make clean
		make
		mv gblur ./build/gblur_${sm}_${bsize}
	done
done

