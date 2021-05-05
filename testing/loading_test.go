package testing

import (
	"fmt"
	math2 "github.com/TrizlyBear/PWS/math"
	"github.com/TrizlyBear/PWS/sequential"
	"github.com/TrizlyBear/PWS/sequential/activation"
	"github.com/TrizlyBear/PWS/sequential/layers"
	"testing"
)

const (
	fps              = 60
	stepSize float64 = 1.0 / (float64(fps) * 4.0)
	padding          = 2
	maxWidth         = 80
)

func TestLoading(t *testing.T) {
	XORx := [][][]float64{{{0.0, 0.0}}, {{0.0, 1.0}}, {{1.0, 0.0}}, {{1.0, 1.0}}}
	XORy := [][][]float64{{{0.0}}, {{1.0}}, {{1.0}}, {{0.0}}}
	XORy = math2.Marge(XORy)
	fmt.Println(XORy)
	n := &sequential.Model{Layers: []sequential.Layer{&layers.FC{Out: 1000},&activation.Tanh{}, &layers.FC{Out: 1},&activation.Tanh{}}}
	n.Fit(XORx, XORy, 10000, 0.1)
}