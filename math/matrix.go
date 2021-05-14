package math

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

func Rand(y int, x int) [][]float64 {
	out := make([][]float64, int(y))
	for i := 0; i < x; i++ {
		for m := 0; m < y; m++ {
			out[m] = append(out[m], rand.Float64()-0.5)
		}
	}
	return out
}

func Dot(x [][]float64, y [][]float64) (r [][]float64, err error) {
	if len(x[0]) != len(y) {
		//fmt.Println(x,y)
		return nil, errors.New("Dimensions not valid %v %v")
	}

	out := make([][]float64, len(x))
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(y[0]); j++ {
			out[i] = append(out[i], 0)
			for k := 0; k < len(x[0]); k++ {
				out[i][j] += (x[i][k] * y[k][j])
			}
		}
	}

	return out, nil
}

func Transpose(slice [][]float64) [][]float64 {
	xl := len(slice[0])
	yl := len(slice)
	result := make([][]float64, xl)
	for i := range result {
		result[i] = make([]float64, yl)
	}
	for i := 0; i < xl; i++ {
		for j := 0; j < yl; j++ {
			result[i][j] = slice[j][i]
		}
	}
	return result
}

func Resize(matrix [][]float64, y int, x int) [][]float64 {
	out := make([][]float64,y)
	for i := range out {
		out[i] = make([]float64, x)
	}

	flat := []float64{}
	for _, i := range matrix {
		for _, o := range i {
			flat = append(flat, o)
		}

	}
	//fmt.Print(flat)
	for i := 0; i < y; i++ {
		for o := 0; o < x; o++ {
			//fmt.Println("AAAAAAAAAA",out, flat)
			out[i][o] = flat[0]
			flat = flat[1:]
		}
	}
	return out
}

func Normalize(in [][][]float64) [][][]float64 {
	smallest := math.Inf(1)
	biggest := math.Inf(-1)
	for _, a := range in {
		for _, b := range a {
			for _, c := range b {
				if c < smallest {
					smallest = c
				}
			}
		}
	}
	for x, a := range in {
		for y, b := range a {
			for z, _ := range b {
				in[x][y][z] -= smallest
			}
		}
	}
	for _, a := range in {
		for _, b := range a {
			for _, c := range b {
				if c > biggest {
					biggest = c
				}
			}
		}
	}
	for x, a := range in {
		for y, b := range a {
			for z, _ := range b {
				in[x][y][z] *= 1 / biggest
			}
		}
	}
	return in
}

func Closest(pred [][]float64, all [][][]float64, truevalue [][]float64) float64 {
	in := false
	for _, x := range all {
		checkcurrent := true
		for y, a := range x {
			for z, _ := range a {
				if truevalue[y][z] != x[y][z] {
					checkcurrent = false
				}
			}

		}
		if checkcurrent {
			in = true
		}
	}
	if !in {
		fmt.Printf("True value not in all values #{lol}")
		return 0
	}

	smallest := float64(math.Inf(1))
	smallesti := -1

	for i, z := range all {
		total := 0.0
		for y, a := range z {
			for x, _ := range a {
				total += math.Abs(z[y][x] - pred[y][x])
			}
		}

		if total < smallest {
			smallest = total
			smallesti = i
		}
	}

	isTrue := true
	for y, z := range all[smallesti] {

		for x, _ := range z {
			if all[smallesti][y][x] != truevalue[y][x] {
				isTrue = false
			}
		}
	}

	if isTrue {
		return 1
	} else {
		return 0
	}
}

func MatriSubs(m1 [][]float64, m2 [][]float64) ([][]float64, error) {
	if len(m1) != len(m2) || len(m1[0]) != len(m2[0]) {
		//fmt.Println(len(m1),len(m2),len(m1[0]),len(m2[0]))
		return nil, errors.New("Dimensions do not match. "+string(len(m1))+" is not "+string(len(m2))+" or "+string(len(m1[0]))+" is not "+string(len(m2[0])))
	}
	out := [][]float64{}
	for y,ely := range m1 {
		row := []float64{}
		for x,_ := range ely {
			row = append(row, m1[y][x] - m2[y][x])
		}
		out = append(out, row)
	}

	return out, nil
}

// Stacks two matrices horizontally
func VertStack(x [][]float64, y [][]float64) [][]float64 {
	for _,row := range y {
		x = append(x, row)
	}
	return x
}

func HorzStack(x [][]float64, y [][]float64) [][]float64 {
	for i,_ := range y {
		for ix,_ := range y[i] {
			x[i] = append(x[i], y[i][ix])
		}
	}
	return x
}

func Zeros(x int, y int) (out [][]float64) {
	for i := 0; i < y; i++ {
		row := []float64{}
		for o := 0; o < x; o++ {
			row = append(row, 0)
		}
		out = append(out, row)
	}
	return out
}