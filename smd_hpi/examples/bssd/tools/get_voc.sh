#!/usr/bin/env bash

# make a data folder
if ! [ -e data ]
then
    mkdir -p data/train
	mkdir -p data/test
else
    if ! [ -e data/train ]
	then
	    mkdir -p data/train
		mkdir -p data/test
	fi
fi

pushd data

voc2007_train="VOCtrainval_06-Nov-2007.tar"
if ! [ -e $voc2007_train ]
then
    echo $voc2007_train "not found, downloading"
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/$voc2007_train
fi
tar -xf $voc2007_train -C train/

voc2007_test="VOCtest_06-Nov-2007.tar"
if ! [ -e $voc2007_test ]
then
    echo $voc2007_test "not found, downloading"
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/$voc2007_test
fi
tar -xf $voc2007_test -C test/

voc2012="VOCtrainval_11-May-2012.tar"
if ! [ -e $voc2012 ]
then
    echo $voc2012 "not found, downloading"
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/$voc2012
fi
tar -xf $voc2012 -C train/

popd
