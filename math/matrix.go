package math

import (
	"errors"
	"math"
	"math/rand"
	"reflect"
)

// Creates a matrix with random values
func Rand2D(y int, x int) [][]float64 {
	out := make([][]float64, int(y))
	for i := 0; i < x; i++ {
		for m := 0; m < y; m++ {
			out[m] = append(out[m], rand.Float64()-0.5)
		}
	}
	return out
}

// Multiplies two 2-dimensional matrices
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

// Determines the correct dimensions of the matrix.
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

// Resizes the matrix
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

	for i := 0; i < y; i++ {
		for o := 0; o < x; o++ {
			out[i][o] = flat[0]
			flat = flat[1:]
		}
	}
	return out
}

// Normalizes the values between <0,1>
func Normalize(in [][][][]float64) [][][][]float64 {
	smallest := math.Inf(1)
	biggest := math.Inf(-1)
	for _, a := range in {
		for _, b := range a {
			for _, c := range b {
				for _,d := range c {
					if d < smallest {
						smallest = d
					}
				}
			}
		}
	}
	for x, a := range in {
		for y, b := range a {
			for z, c := range b {
				for omega,_ := range c {
					in[x][y][z][omega] -= smallest
				}
			}
		}
	}
	
	for _, a := range in {
		for _, b := range a {
			for _, c := range b {
				for _,d := range c {
					if d > biggest {
						biggest = d
					}
				}
			}
		}
	}
	for x, a := range in {
		for y, b := range a {
			for z, c := range b {
				for omega,_ := range c {
					in[x][y][z][omega] *= 1 / biggest
				}
			}
		}
	}
	return in
}

// Checks if the predicted value is the closest to the true value, returns 1 if true, 0 if false
func Closest(pred [][][]float64, all [][][][]float64, truevalue [][][]float64) float64 {
	/*in := false
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
	*/
	
	smallest := float64(math.Inf(1))
	smallesti := -1

	for i, el := range all {
		total := 0.0
		for z,dim := range el {
			for y, a := range dim {
				for x, _ := range a {
					total += math.Abs(el[z][y][x] - pred[z][y][x])
				}
			}
		}

		if total < smallest {
			smallest = total
			smallesti = i
		}
	}
	isTrue := true
	for y, dim := range all[smallesti] {

		for x, row := range dim {
			for z,_ := range row {
				if all[smallesti][y][x][z] != truevalue[y][x][z] {
					isTrue = false
				}
			}
		}
	}
	
	if isTrue {
		return 1
	} else {
		return 0
	}
}

// Substracts two matrices
func MatSub(m1 [][]float64, m2 [][]float64) ([][]float64, error) {
	if len(m1) != len(m2) || len(m1[0]) != len(m2[0]) {
		// Returns error if dimensions do not match
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

// Adds two matrices
func MatAdd(m1 [][]float64, m2 [][]float64) ([][]float64, error) {
	if len(m1) != len(m2) || len(m1[0]) != len(m2[0]) {
		// Returns error if dimensions do not match
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

// Stacks two matrices vertically
func VertStack(x [][]float64, y [][]float64) [][]float64 {
	for _,row := range y {
		x = append(x, row)
	}
	return x
}

// Stacks two matrices horizontally
func HorzStack(x [][]float64, y [][]float64) [][]float64 {
	out := [][]float64{}
	for _,row := range x {
		out = append(out, row)
	}
	for i,_ := range y {
		for ix,_ := range y[i] {
			out[i] = append(out[i], y[i][ix])
		}
	}
	return out
}
// Creates a matrix of zeros with dimentions x and y
func Zeros2D(x int, y int) (out [][]float64) {
	for i := 0; i < y; i++ {
		row := []float64{}
		for o := 0; o < x; o++ {
			row = append(row, 0)
		}
		out = append(out, row)
	}
	return out
}

// Returns the average value of a matrix
func MatMean(m [][]float64) float64 {
	vals := []float64{}
	for _,r := range m {
		vals = append(vals, Mean(r))
	}
	return Mean(vals)
}

// Defines MatOp function
type MatOpF func(val float64, x int, y int) float64

// Applies a given function on a matrix
func MatOp(m [][]float64, f MatOpF) [][]float64 {
	for y,_ := range m {
		for x,_ := range m[y] {
			m[y][x] = f(m[y][x], x, y)
		}
	}
	return m
}

// Calculates the sum of a matrix
func MatSum(m [][]float64) (out float64) {
	for y,_ := range m {
		for x,_ := range m[y] {
			out += m[y][x]
		}
	}
	return out
}

// Generate a multidimensional matrix with random numbers
func Rand(seq ...int) interface{} {
	// Declare numbers to fill
	nums := seq[0]
	for _,el := range seq[1:] {
		nums *= el
	}

	// Generate the sequence of numbers
	left := []reflect.Value{}
	for i := 0; i < nums; i++ {
		left = append(left, reflect.ValueOf(rand.Float64() - .5))
	}

	// Declare the type template
	t := reflect.TypeOf([]float64{})

	// Reverse the sequence
	for i, j := 0, len(seq)-1; i < j; i, j = i+1, j-1 {
		seq[i], seq[j] = seq[j], seq[i]
	}

	// Loop over the sequence
	for _,a := range seq {
		n := []reflect.Value{}
		for y := 0; y < nums / a; y++ {
			holder := reflect.Zero(t)
			for i := 0; i < a; i++ {
				holder = reflect.Append(holder, left[0])
				left = left[1:]
			}
			n = append(n, holder)
		}
		nums /= a
		left = n
		t = reflect.SliceOf(t)
	}
	return left[0].Interface()
}

// Generate a multidimensional matrix with zeros
func Zeros(seq ...int) interface{} {
	// Declare numbers to fill
	nums := seq[0]
	for _,el := range seq[1:] {
		nums *= el
	}

	// Generate the sequence of numbers
	left := []reflect.Value{}
	for i := 0; i < nums; i++ {
		left = append(left, reflect.ValueOf(float64(0)))
	}

	// Declare the type template
	t := reflect.TypeOf([]float64{})

	// Reverse the sequence
	for i, j := 0, len(seq)-1; i < j; i, j = i+1, j-1 {
		seq[i], seq[j] = seq[j], seq[i]
	}

	// Loop over the sequence
	for _,a := range seq {
		n := []reflect.Value{}
		for y := 0; y < nums / a; y++ {
			holder := reflect.Zero(t)
			for i := 0; i < a; i++ {
				holder = reflect.Append(holder, left[0])
				left = left[1:]
			}
			n = append(n, holder)
		}
		nums /= a
		left = n
		t = reflect.SliceOf(t)
	}
	return left[0].Interface()
}

// Adds padding to a matrix
func Padding(in [][]float64, s int) [][]float64 {
	if s == 0 {
		return in
	}
	dimX := len(in[0])
	dimY := len(in)
	sidefill := Zeros(dimY,s).([][]float64)
	topfill := Zeros(s,dimX + (2 * s)).([][]float64)
	return VertStack(VertStack(topfill,HorzStack(sidefill,HorzStack(in,sidefill))),topfill)
}

func OutIndex(in [][][]float64) int {
	for i,_ := range in {
		if in[i][0][0] == 1 {
			return i
		}
	}

	return -1
}

func Rotate180(in [][]float64) [][]float64{
	out := in

	for y,_ := range out {
		for x,_ := range out[y][:len(out[y])/2] {
			out[y][x],out[y][len(out[y])-x-1]=out[y][len(out[y])-x-1],out[y][x]
		}
	}
	for y,_ := range out[:len(out)/2] {
		out[y],out[len(out)-y-1] = out[len(out)-y-1],out[y]
	}
	return out
}