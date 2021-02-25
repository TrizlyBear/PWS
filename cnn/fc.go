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

func (e *FC) Forward(in [][]float64, size dim) [][]float64 {
	if e.init != true {

		(*e).weights = math.Rand(e.Out*size.x, size.y)
		fmt.Println(e.weights)
		(*e).bias = math.Rand(1, e.Out)
		(*e).init = true
	}

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
	//fmt.Println(err)
	ierr := math.Transpose(e.weights)

	/*if er != nil {
		fmt.Println(er)
	}*/
	// Input error
	if len(err) == 1 && len(err[0]) == 1 {
		for y, _ := range ierr {
			for x, _ := range ierr[y] {
				ierr[y][x] *= err[0][0]
			}
		}
	} else {
		//err = math.Transpose(err)
		ierr, _ = math.Dot(err, ierr)
		_, er := math.Dot(err, ierr)
		if er != nil {
			//fmt.Println("ER",len(err),len(err[0]),"WE",len(e.weights),len(e.weights[0]))
			//fmt.Println(er)
		}
	}

	//_ = er

	werr := math.Transpose(ierr)

	// Calculate weights error
	if len(err) == 1 && len(err[0]) == 1 {
		for y, _ := range werr {
			for x, _ := range werr[y] {
				werr[y][x] *= err[0][0]
			}
		}
	} else {
		werr, _ = math.Dot(werr, err)
	}

	// Update bias
	for x, _ := range e.bias {
		e.bias[x][0] -= lr * err[0][x]
		//fmt.Println(err,e.bias)
	}

	//werr = math.Transpose(werr)
	for y, _ := range e.weights {
		for x, _ := range e.weights[0] {
			e.weights[y][x] -= lr * werr[y][x]
		}
	}

	//fmt.Println(ierr)

	return ierr
}
