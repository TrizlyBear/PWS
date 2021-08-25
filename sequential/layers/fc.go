package layers

import (
	"fmt"
	"github.com/TrizlyBear/PWS/math"
)

// Fully Connected layer
type FC struct {
	Out     int
	weights [][]float64
	bias    []float64
	init    bool
	input	[][][]float64
}

// FC Forward function
func (e *FC) Forward(in [][][]float64)  (out [][][]float64) {
	conv := []float64{}

	for _,x := range in {
		conv = append(conv, x[0][0])
	}

	// Initialize weights and biases if there are none
	if e.init != true {
		(*e).weights = math.Rand2D(len(conv), e.Out)
		(*e).bias = math.Rand2D(1, e.Out)[0]
		(*e).init = true
	}

	// Save input for later use
	(*e).input = in



	// Multiply the weights and input for output
	nonconvout, er := math.Dot([][]float64{conv}, e.weights)
	if er != nil {
		fmt.Println(er)
	}

	for i,_ := range nonconvout {
		nonconvout[0][i] += (*e).bias[i]
	}

	for _,e := range nonconvout[0] {
		out = append(out, [][]float64{{e}})
	}

	return out
}

func (e FC) Backward(err [][][]float64, lr float64) [][][]float64 {
	//fmt.Println(err)
	rerr := [][]float64{{}}

	for _,x := range err {
		rerr[0] = append(rerr[0], x[0][0])
	}

	rinput := [][]float64{{}}

	for _,x := range e.input {
		rinput[0] = append(rinput[0], x[0][0])
	}

	// Calculate the error for the next layer
	rierr, er := math.Dot(rerr,math.Transpose(e.weights))

	if er != nil {
		fmt.Println(er)
	}

	// Calculate the error of the weights
	werr, er := math.Dot(math.Transpose(rinput), rerr)

	if er != nil {
		fmt.Print(er)
	}

	// Adjust the bias according to the error
	//fmt.Println(rerr)
	for x, _ := range e.bias {
		e.bias[x] -= lr * rerr[0][x]
	}

	// Adjust the weights by substracting it weight error multiplied by the learning rate
	for y, _ := range e.weights {
		for x, _ := range e.weights[0] {
			e.weights[y][x] -= lr * werr[y][x]
		}
	}

	ierr := [][][]float64{}

	for _,e := range rierr[0] {
		ierr = append(ierr, [][]float64{{e}})
	}

	// Return the error for the next layer
	return ierr
}