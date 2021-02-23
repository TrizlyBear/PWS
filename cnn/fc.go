package cnn

import (
	"fmt"
	"github.com/TrizlyBear/PWS/math"
)

type FC struct {
	Out     int
	weights [][]float64
	bias    [][]float64
	init    bool
}

func (e *FC) Forward(in [][]float64) [][]float64 {
	if e.init != true {
		(*e).weights = math.Rand(len(in), e.Out)
		(*e).bias = math.Rand(1, e.Out)
		(*e).init = true
	}

	out, _ := math.Dot(e.weights, in)

	for y, _ := range out {
		for x, _ := range out[y] {
			out[y][x] += (*e).bias[y][0]
		}
	}
	return out
}

func (e *FC) Backward(err [][]float64, lr float64) [][]float64 {
	fmt.Println("Bweights", (*e).weights)
	ierr := math.Transpose((*e).weights)
	/*if er != nil {
		fmt.Println(er)
	}*/
	// Input error
	ierr, _ = math.Dot(err, ierr)
	/*for y,_ := range ierr {
		for x,_ := range ierr[y] {
			ierr[y][x] *= err
		}
	}*/

	werr := math.Transpose(ierr)

	// Calculate weights error
	werr, _ = math.Dot(werr, err)
	/*for y,_ := range werr {
		for x,_ := range werr[y] {
			werr[y][x] *= err
		}
	}*/

	// Update bias
	for x, _ := range e.bias {
		e.bias[x][0] -= lr * err[0][x]
	}

	for y, _ := range e.weights {
		for x, _ := range e.weights {
			e.weights[y][x] -= lr * werr[y][x]
		}
	}

	return ierr
}
