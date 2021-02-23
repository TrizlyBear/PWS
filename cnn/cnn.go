package cnn

import (
	"fmt"
	math2 "github.com/TrizlyBear/PWS/math"
	"math"
)

type Cnn struct {
	Layers []interface{}
}

func (e Cnn) forward(in [][]float64) [][]float64 {
	for _, el := range e.Layers {
		if layer, ok := el.(*FC); ok {
			in = (*(&layer)).Forward(in)
		}
		if layer, ok := el.(*Flatten); ok {
			in, _ = (*(&layer)).Forward(in)
		}
	}
	return in
}

func (e Cnn) backward(err [][]float64, lr float64) [][]float64 {
	reversed := math2.Reverse(e.Layers)
	for _, el := range reversed {
		if layer, ok := el.(*FC); ok {
			err = (*(&layer)).Backward(err, lr)
		}
		if layer, ok := el.(Flatten); ok {
			err = (*(&layer)).Backward(err)
		}
	}
	reversed = math2.Reverse(reversed)
	return err
}

func (e Cnn) Fit(x_train [][][]float64, y_train [][][]float64, epochs int, lr float64) {
	for i := 1; i < epochs+1; i++ {
		err := 0.0
		for ie, x := range x_train {
			//fmt.Println("lol")
			out := e.forward(x)
			//fmt.Println("out","object",ie)
			err += math.Pow(y_train[ie][0][0]-out[0][0], 2)
			e.backward([][]float64{{2 * (out[0][0] - y_train[ie][0][0])}}, lr)
		}
		fmt.Println("Epoch", i, "Error", err)
	}
}
