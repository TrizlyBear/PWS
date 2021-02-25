package main

import (
	"github.com/TrizlyBear/PWS/cnn"
)

func main() {
	XORx := [][][]float64{{{0.0, 0.0}}, {{0.0, 1.0}}, {{1.0, 0.0}}, {{1.0, 1.0}}}
	XORy := [][][]float64{{{0.0}}, {{1.0}}, {{1.0}}, {{0.0}}}
	n := &cnn.Cnn{Layers: []interface{}{&cnn.FC{Out: 3}, &cnn.Flatten{}, &cnn.FC{Out: 1}}}
	n.Fit(XORx, XORy, 10, 0.1)

	//test := &cnn.FC{Out: 10}
	//fmt.Println(test.Forward(XORx[1]))
}
