package testing

import (
	"fmt"
	dataset "github.com/TrizlyBear/PWS"
	"testing"
)

func TestDataset(t *testing.T) {
	ds, err := dataset.FromCSV("../datasets/mnist_train.csv",[]int{0},dataset.Max(2))
	if err != nil {
		panic(err)
	}
	ds.Reshape(28,28)
	fmt.Println(ds.Split(1))
}
