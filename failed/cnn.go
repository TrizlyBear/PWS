package failed

import (
	"PWS/cnn"
	"fmt"
	math2 "github.com/TrizlyBear/PWS/math"
	"math"
)

type Model struct {
	Name     string
	Layers   []interface{}
	Filters  []float64 `json: "omitempty"`
	Features []string  `json: "omitempty"`
}

type Layer interface{}

func (e Model) Forward(in [][]float64) (output [][]float64, result []float64, err error) {
	current := in
	//&Model{Filters: []}
	for _, x := range e.Layers {
		if layer, ok := x.(cnn.MaxPooling); ok {
			out, err := layer.Forward(current)
			if err != nil {
				return nil, nil, err
			} else {
				current = out
			}
		}
		if layer, ok := x.(cnn.AvgPooling); ok {
			out, err := layer.Forward(current)
			if err != nil {
				return nil, nil, err
			} else {
				current = out
			}
		}
		if layer, ok := x.(*Flatten); ok {
			out, err := (*(&layer)).Forward(current)
			if err != nil {
				return nil, nil, err
			} else {
				current = [][]float64{out}
			}
		}
		if layer, ok := x.(*ReLu); ok {
			out, err := (*(&layer)).Forward(current)
			if err != nil {
				return nil, nil, err
			} else {
				current = out
			}
		}
		if layer, ok := x.(Conv2D); ok {
			out, err := layer.Forward(current, []float64{1, 0, 1, 0})
			if err != nil {
				return nil, nil, err
			} else {
				current = out
			}
		}
		if layer, ok := x.(*FC); ok {
			fmt.Println("forward", current)
			out, err := (*(&layer)).Forward(current, (*layer).In, (*layer).Out)
			if err != nil {
				return nil, nil, err
			} else {
				//fmt.Println(out)
				current = out
			}
		}
	}

	//MaxPooling{e.Config[0].(int), e.Config[1].(int)}.Forward(in)
	return current, nil, nil
}

func (e Model) Backward(error float64, lr float64) {
	err := error
	for _, x := range math2.Reverse(e.Layers) {
		if layer, ok := x.(*FC); ok {
			var test = (*(&layer)).Backward(err, lr)
			err = test[0][0]
		}
		/*if layer, ok := x.(Flatten); ok {
			//out, err := layer.Backward()
		}*/
	}
}

func (e Model) Predict(in [][]float64) string {
	var biggest float64 = 0
	var biggesti = -1
	_, output, _ := e.Forward(in)
	for i, el := range output {
		if el > biggest {

			biggest, biggesti = el, i
		}
	}

	return e.Features[biggesti]
}

func (e Model) Fit(x_train [][][]float64, y_train [][]float64, x_valid [][][]float64, y_valid []float64, epochs int, learning_rate float64) {
	if len(x_train) != len(y_train) || len(x_valid) != len(y_valid) {
		fmt.Print("! ", "Train or Validation datasets dont have same amount data as predictable features")
		return
	}
	var err float64 = 0
	for i := 0; i < epochs; i++ {
		for ie, el := range x_train {
			fmt.Println("Next train Before:", el)
			//fmt.Println(el)
			out, _, _ := e.Forward(el)
			fmt.Println("After", out)
			//fmt.Println(out)
			err += math.Pow(y_train[ie][0]-out[0][0], 2)
			e.Backward(2*(out[0][0]-y_train[ie][0]), learning_rate)
			_ = math.Abs(1.0)
			_ = ie
			//flat := Flatten{}
			//fmt.Println(flat.Forward(el))
		}
		fmt.Println("Epoch:", i+1, "| Accuracy", "| Test accuracy", "| Error", err)
		err = 0
	}
}
