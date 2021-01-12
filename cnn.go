package cnn

import "fmt"

type Model struct {
	Name     string
	Layers   []interface{}
	Filters  []float64 `json: "omitempty"`
	Features []string  `json: "omitempty"`
}

type TypeName []interface{}
type Layer struct {
	L []interface{}
}

func (e Model) Forward(in [][]float64) (output [][]float64, result []float64, err error) {
	current := in
	//&Model{Filters: []}
	for _, x := range e.Layers {
		if layer, ok := x.(MaxPooling); ok {
			out, err := layer.Forward(current)
			if err != nil {
				return nil, nil, err
			} else {
				current = out
			}
		}
		if layer, ok := x.(AvgPooling); ok {
			out, err := layer.Forward(current)
			if err != nil {
				return nil, nil, err
			} else {
				current = out
			}
		}
		if layer, ok := x.(Flatten); ok {
			out, err := layer.Forward(current)
			if err != nil {
				return nil, nil, err
			} else {
				current = [][]float64{out}
			}
		}
		if layer, ok := x.(ReLu); ok {
			out, err := layer.Forward(current)
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
	}

	//MaxPooling{e.Config[0].(int), e.Config[1].(int)}.Forward(in)
	return current, nil, nil
}

func (e Model) Backward() {

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

func (e Model) Fit(x_train [][][]float64, y_train []string, x_valid [][][]float64, y_valid []string, epochs int, learning_rate float64) {
	if len(x_train) != len(y_train) || len(x_valid) != len(y_valid) {
		fmt.Print("! ", "Train or Validation datasets dont have same amount data as predictable features")
		return
	}
	for i := 0; i < epochs; i++ {
		for _, el := range x_train {
			e.Forward(el)
			e.Backward()
		}
		for i, el := range x_valid {
			/*if e.Forward(el).(float64) != y_valid[i] {

			}*/
			var _, _ = i, el
		}
		fmt.Println("Epoch:", i+1, "| Accuracy", "| Test accuracy")
	}
}
