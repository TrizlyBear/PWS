package main

import (
"github.com/TrizlyBear/PWS/sequential"
"github.com/TrizlyBear/PWS/sequential/activation"
"github.com/TrizlyBear/PWS/sequential/layers"
)

func main() {
	XORx := [][][]float64{{{0.0, 0.0}}, {{0.0, 1.0}}, {{1.0, 0.0}}, {{1.0, 1.0}}}
	XORy := [][][]float64{{{0.0}}, {{1.0}}, {{1.0}}, {{0.0}}}
	n := &sequential.Cnn{Layers: []sequential.Layer{&layers.FC{Out: 10},&activation.Tanh{}, &layers.FC{Out: 1},&activation.Tanh{}}}
	n.Fit(XORx, XORy, 1000, 0.1)
	n.Save("./models")
}

