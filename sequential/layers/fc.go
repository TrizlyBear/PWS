package layers

import (
	"fmt"
	"github.com/TrizlyBear/PWS/math"
)

type FC struct {
	Out     int
	weights [][]float64
	bias    [][]float64
	init    bool
	input	[][]float64
}

func (e *FC) Forward(in [][]float64) [][]float64 {
	if e.init != true {
		(*e).weights = math.Rand(len(in[0]), e.Out*len(in))
		(*e).bias = math.Rand(1, e.Out)
		(*e).init = true
	}

	(*e).input = in

	out, er := math.Dot(in, e.weights)
	if er != nil {
		fmt.Println(er)
	}
	for y, _ := range out {
		for x, _ := range out[y] {
			out[y][x] += (*e).bias[y][0]
		}
	}

	return out
}

func (e FC) Backward(err [][]float64, lr float64) [][]float64 {

	ierr, er := math.Dot(err,math.Transpose(e.weights))

	if er != nil {
		fmt.Println(er)
	}

	werr, er := math.Dot(math.Transpose(e.input), err)

	if er != nil {
		fmt.Print(er)
	}

	for x, _ := range e.bias {
		e.bias[x][0] -= lr * err[0][x]
	}

	for y, _ := range e.weights {
		for x, _ := range e.weights[0] {
			e.weights[y][x] -= lr * werr[y][x]
		}
	}

	return ierr
}
