package main

import (
	"fmt"
	"github.com/TrizlyBear/PWS/cnn"
	"github.com/TrizlyBear/PWS/cnn/activation"
)

func main() {
	res := []float64{}
	for _,_ = range []float64{0:0} {
		XORx := [][][]float64{{{0.0, 0.0}}, {{0.0, 1.0}}, {{1.0, 0.0}}, {{1.0, 1.0}}}
		XORy := [][][]float64{{{0.0}}, {{1.0}}, {{1.0}}, {{0.0}}}
		n := &cnn.Cnn{Layers: []cnn.Layer{&cnn.FC{Out: 10},&activation.Tanh{}, &cnn.FC{Out: 1}, &activation.Tanh{}}}
		res = append(res,n.Fit(XORx, XORy, 10000, 0.1).Accuracy)
	}
	fmt.Println(res)
}
