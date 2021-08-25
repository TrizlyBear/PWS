package testing

import (
	"github.com/TrizlyBear/PWS/sequential"
	"github.com/TrizlyBear/PWS/sequential/activation"
	"github.com/TrizlyBear/PWS/sequential/layers"
	"testing"
)

// Run a model on XOR operation truth table
func TestXOR(t *testing.T) {
	XORx := [][][][]float64{
		{{{0.0}}, {{0.0}}},
		{{{0.0}}, {{1.0}}},
		{{{1.0}}, {{0.0}}},
		{{{1.0}}, {{1.0}}},
	}
	XORy := [][][][]float64{{{{0.0}}}, {{{1.0}}}, {{{1.0}}}, {{{0.0}}}}
	n := &sequential.Model{Layers: []sequential.Layer{&layers.FC{Out: 10},&activation.Tanh{}, &layers.FC{Out: 1},&activation.Tanh{}}}
	n.Fit(XORx, XORy, 1000, 0.1)
}
