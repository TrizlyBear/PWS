package failed

import (
	"fmt"
	math2 "github.com/TrizlyBear/PWS/math"
	"math"
)

type Flatten struct {
}

type ReLu struct {
}

type FC struct {
	Biases  [][]float64
	Weights [][]float64
	Input   [][]float64
	In      int
	Out     int
}

func (e ReLu) Forward(in [][]float64) ([][]float64, error) {
	out := in
	for x, l := range in {
		for y, i := range l {
			out[x][y] = math.Max(0, i)
		}
	}
	return out, nil
}

func (e *FC) Forward(in [][]float64, out int, insize int) ([][]float64, error) {
	fmt.Println("Forward")
	if len(e.Biases) == 0 {
		(*e).Biases = math2.Rand(1, 1)
		if (*e).Out == 1 {
			(*e).Weights = math2.Rand((*e).Out, (*e).In*(*e).In)
		} else {
			(*e).Weights = math2.Rand((*e).Out, (*e).In)
		}

		if (*e).Out == 1 {

			if len(in) > 1 {
				idk, _ := Flatten{}.Forward(in)
				in = [][]float64{idk}
				(*e).Input = in
			}
		} else {
			(*e).Input = in
		}

	}
	if (*e).Out == 1 {

		if len(in) > 1 {
			idk, _ := Flatten{}.Forward(in)
			in = [][]float64{idk}
		}
	}

	//fmt.Println("INWEIGHT" ,Transpose(in),Transpose(e.Weights))
	/*output := [][]float64{}
	if (*e).Out != 1 {*/
	output, err := math2.Dot(math2.Transpose(in), math2.Transpose(e.Weights))
	/*} else {
		var idk []float64
		for _,e := range in {
			idk = append(idk, e*)
		}
	}*/

	if err != nil {
		fmt.Println(err.Error(), "Poop")
	}
	for y, el := range output {
		for x, _ := range el {
			output[y][x] += e.Biases[0][0]
		}
	}

	return output, nil
}

func (e *FC) Backward(error float64, lr float64) [][]float64 {
	fmt.Println("Backward")
	inputerr := (*e).Weights
	for y, el := range inputerr {
		for x, _ := range el {
			inputerr[y][x] *= error

		}
	}

	//weighterr,_ := Dot( Transpose((*e).Input),[][]float64{{error}},)
	weighterr := (*e).Input

	for y, el := range weighterr {
		for x, _ := range el {
			weighterr[y][x] *= error

		}
	}

	for y, el := range e.Weights {
		for x, _ := range el {
			//fmt.Println("ERROR",(*e).Weights,weighterr)
			(*e).Weights[y][x] -= math2.Transpose(weighterr)[y][x] * lr
		}
	}

	for y, el := range e.Biases {
		for x, _ := range el {
			(*e).Biases[y][x] -= (lr * error)
		}
	}
	//fmt.Println("INERR",inputerr)
	//e.Weights -= weighterr * lr
	//e.Biases -= lr * error
	return [][]float64{{math2.Sum(inputerr[0])}}
}
