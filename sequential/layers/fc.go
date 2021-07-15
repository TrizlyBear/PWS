package layers

import (
	"fmt"
	"github.com/TrizlyBear/PWS/math"
)

// Fully Connected layer
type FC struct {
	Out     int
	weights [][]float64
	bias    [][]float64
	init    bool
	input	[][]float64
}

// FC Forward function
func (e *FC) Forward(in [][]float64) [][]float64 {
	// Initialize weights and biases if there are none
	if e.init != true {
		(*e).weights = math.Rand(len(in[0]), e.Out*len(in))
		(*e).bias = math.Rand(1, e.Out)
		(*e).init = true
	}

	// Save input for later use
	(*e).input = in

	// Multiply the weights and input for output
	out, er := math.Dot(in, e.weights)
	if er != nil {
		fmt.Println(er)
	}

	// Apply the baises to the output
	for y, _ := range out {
		for x, _ := range out[y] {
			out[y][x] += (*e).bias[y][0]
		}
	}

	return out
}

func (e FC) Backward(err [][]float64, lr float64) [][]float64 {

	// Calculate the error for the next layer
	ierr, er := math.Dot(err,math.Transpose(e.weights))

	if er != nil {
		fmt.Println(er)
	}

	// Calculate the error of the weights
	werr, er := math.Dot(math.Transpose(e.input), err)

	if er != nil {
		fmt.Print(er)
	}

	// Adjust the bias according to the error
	for x, _ := range e.bias {
		e.bias[x][0] -= lr * err[0][x]
	}

	// Adjust the weights by substracting it weight error multiplied by the learning rate
	for y, _ := range e.weights {
		for x, _ := range e.weights[0] {
			e.weights[y][x] -= lr * werr[y][x]
		}
	}

	// Return the error for the next layer
	return ierr
}
