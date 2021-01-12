package main

import "github.com/TrizlyBear/PWS"

func main() {
	m := cnn.Model{Name: "Test", Layers: []interface{}{cnn.Conv2D{
		Ksize:      2,
		Stride:     1,
		Neurons:    5,
		Activation: "relu",
	}, cnn.MaxPooling{2, 1}}}
	m.Fit([][][]float64{{{9, 9, 9, 9}, {9, 9, 9, 9}, {9, 9, 9, 9}, {9, 9, 9, 9}}}, []string{"yes"}, [][][]float64{{{9, 9, 9, 9}, {9, 9, 9, 9}, {9, 9, 9, 9}, {9, 9, 9, 9}}}, []string{"POO"}, 30, 0.1)
}
