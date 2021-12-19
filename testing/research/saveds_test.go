package research

import (
	dataset2 "github.com/TrizlyBear/PWS/utils/dataset"
	"image/jpeg"
	"testing"
)

func TestSaveFolder(t *testing.T) {
	set, err := dataset2.FromFolder("../../datasets/TYN", jpeg.Decode, 150, 150, dataset2.LabelToIndex(2) )
	if err != nil {
		panic(err)
	}

	err = dataset2.SaveDS(set,"../../datasets/tumoryn.bin")
	if err != nil {
		panic(err)
	}
}
