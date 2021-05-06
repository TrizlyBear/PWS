package dataset

import (
	"encoding/csv"
	"errors"
	"github.com/TrizlyBear/PWS/math"
	math2 "math"
	"os"
	"reflect"
	"strconv"
)

type datasetOption func(*Dataset)

type Dataset struct {
	Name 	string
	X 		[][][]float64
	Y 		[][][]float64
}

func inlist(list interface{}, item interface{}) bool {
	listVal := reflect.ValueOf(list)
	for i := 0; i < listVal.Len(); i++ {
		if listVal.Index(i).Interface() == item {
			return true
		}
	}
	return false
}

func Max(max int) datasetOption {
	return func(dataset *Dataset) {
		dataset.X = dataset.X[:max]
		dataset.Y = dataset.Y[:max]
	}
}

func LabelToIndex(length int) datasetOption{
	return func(dataset *Dataset) {
		for i,x := range dataset.Y {
			yar := [][]float64{{}}
			for o := 0; o < length; o++ {
				yar[0] = append(yar[0], 0)
			}
			yar[0][int(x[0][0])] = float64(1)
			dataset.Y[i] = yar
		}
	}
}

// Load a dataset from a CSV file
func FromCSV(file string, labels []int, options ...datasetOption) (*Dataset, error) {
	w, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	r, err := csv.NewReader(w).ReadAll()
	if err != nil {
		panic(err)
	}

	set := &Dataset{}

	X := [][][]float64{}
	Y := [][][]float64{}

	for _,row := range r[1:] {
		x := [][]float64{{}}
		y := [][]float64{{}}
		for i,el := range row {
			//fmt.Println(row)
			val, err := strconv.ParseFloat(el, 64)
			if err != nil {
				panic(err)
			}
			if inlist(labels, i) {
				y[0] = append(y[0], val)
			} else {
				x[0] = append(x[0], val)
			}
		}
		Y = append(Y, y)
		X = append(X, x)
	}

	set.X = X
	set.Y = Y

	for _,option := range options {
		option(set)
	}

	return set, nil
}

func (ds *Dataset) Reshape(x int,y int)  {
	for i,X := range ds.X {
		ds.X[i] = math.Resize(X,y,x)
	}
}

func (ds *Dataset) Split(train_size float64) ([][][]float64, [][][]float64, [][][]float64, [][][]float64) {
	if train_size < 0 || train_size > 1 {
		panic(errors.New("Training size must be bigger than 0 or smaller than 1"))
	}
	from := int(math2.Round(float64(len(ds.X)) * train_size))

	return ds.X[:from], ds.Y[:from], ds.X[from:], ds.Y[from:]
}
