package research

import (
	dataset2 "github.com/TrizlyBear/PWS/utils/dataset"
	"image/jpeg"
	"testing"
)

func TestSaveFolder(t *testing.T) {
	set, err := dataset2.FromFolder("../../datasets/Training", jpeg.Decode, 50, 50 /*dataset.Resize(100,100)*/)
	if err != nil {
		panic(err)
	}

	err = dataset2.SaveDS(set,"../../datasets/images.bin")
	if err != nil {
		panic(err)
	}
}
