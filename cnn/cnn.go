package cnn

import (
	"fmt"
	math2 "github.com/TrizlyBear/PWS/math"
	"math"
	"time"
)

type Cnn struct {
	Layers []Layer
	Dim    dim
}

type Layer interface {
	Forward([][]float64) [][]float64
	Backward([][]float64, float64) [][]float64
}

type dim struct {
	x int
	y int
}

type Result struct {
	Epochs 		int
	Duration 	time.Duration
	Error 		float64
	Accuracy	float64
}

func reverse(s []Layer) []Layer {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
	return s
}

func (e Cnn) forward(in [][]float64) [][]float64 {
	for _, el := range e.Layers {
		in = (*(&el)).Forward(in)
	}
	return in
}

func (e Cnn) backward(err [][]float64, lr float64) [][]float64 {
	reversed := reverse(e.Layers)
	for _, el := range reversed {
		//fmt.Println("Back")
		if layer, ok := el.(*FC); ok {
			err = (*(&layer)).Backward(err, lr)
		}
		if layer, ok := el.(*Flatten); ok {
			err = (*(&layer)).Backward(err, lr)
		}
	}
	reversed = reverse(reversed)
	return err
}

func (e Cnn) Fit(x_train [][][]float64, y_train [][][]float64, epochs int, lr float64) Result {
	e.Dim.y = len(x_train[0])
	e.Dim.x = len(x_train[0][0])
	start := time.Now()
	res := Result{Epochs: epochs}
	for i := 1; i < epochs+1; i++ {
		err := 0.0
		avg := []float64{}
		for ie, x := range x_train {
			//fmt.Println("lol")
			out := e.forward(x)
			//fmt.Println("out","object",ie)
			err += math.Pow(y_train[ie][0][0]-out[0][0], 2)
			avg = append(avg, (y_train[ie][0][0]-math.Abs(y_train[ie][0][0]-out[0][0]))/y_train[ie][0][0])
			e.backward([][]float64{{2 * (out[0][0] - y_train[ie][0][0])}}, lr)
		}
		acc := math2.Mean(avg)
		fmt.Println("Epoch", i, "Error", err, "Accuracy", acc)
		res.Accuracy = acc
		res.Error = err
	}
	end := time.Since(start)

	res.Duration = end

	return res
}
