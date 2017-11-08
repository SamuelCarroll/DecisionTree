package DecisionTree

import (
	"math"
	"reflect"

	"github.com/SamuelCarroll/DataTypes"
)

func avgClass(allData []*dataTypes.Data) {
	for _, datum := range allData {
		if datum.Class == 1 {
			class1.averages = runningAvg(class1.averages, *datum, class1.count)
			class1.count++
			class1Sample = append(class1Sample, datum)
		} else if datum.Class == 2 {
			class2.averages = runningAvg(class2.averages, *datum, class2.count)
			class2.count++
			class2Sample = append(class2Sample, datum)
		} else if datum.Class == 3 {
			class3.averages = runningAvg(class3.averages, *datum, class3.count)
			class3.count++
			class3Sample = append(class3Sample, datum)
		}
	}
	class1.stdDev = findStds(class1Sample, class1)
	class2.stdDev = findStds(class2Sample, class2)
	class3.stdDev = findStds(class3Sample, class3)
}

func getMajority(data []*dataTypes.Data) int {
	count1, count2, count3 := 0, 0, 0
	for _, datum := range data {
		if datum.Class == 1 {
			count1++
		} else if datum.Class == 2 {
			count2++
		} else {
			count3++
		}
	}
	if count1 > count2 && count1 > count3 {
		return 1
	} else if count2 > count1 && count2 > count3 {
		return 2
	} else {
		return 3
	}
}

func stoppingCond(nodeData []*dataTypes.Data, stopCond float64) bool {
	var count [3]int
	var percent [3]float64

	for _, elem := range nodeData {
		count[elem.Class-1]++
	}

	for i := range count {
		percent[i] = float64(count[i]) / float64(len(nodeData))
		if percent[i] >= stopCond {
			return true
		}
	}

	return false
}

func findEntropy(valueIndex int, avg, stdDev float64, nodeData []*dataTypes.Data) float64 {
	//TODO consider turning this into an array
	in1, in2, in3 := 0.0, 0.0, 0.0
	e1, w1, e2, w2, e3, w3 := 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

	for _, datum := range nodeData {

		instance := getFloatReflectVal(datum.FeatureSlice[valueIndex])
		if datum.Class == 1 {
			in1 += countClass(instance, avg+stdDev)
		} else if datum.Class == 2 {
			in2 += countClass(instance, avg+stdDev)
		} else if datum.Class == 3 {
			in3 += countClass(instance, avg+stdDev)
		}
	}

	lenData := float64(len(nodeData))

	if in1 > 0 {
		w1 = in1 / lenData
		e1 = (w1) * math.Log2(w1)
	}
	if in2 > 0 {
		w2 = in2 / lenData
		e2 = (w2) * math.Log2(w2)
	}
	if in3 > 0 {
		w3 = in3 / lenData
		e3 = (w3) * math.Log2(w3)
	}

	entropy := w1*e1 + w2*e2 + w3*e3

	return entropy * -1
}

func countClass(instance float64, splitVal float64) float64 {
	if instance < splitVal {
		return 1
	}

	return 0
}

func initializeAvgs(example dataTypes.Data) []interface{} {
	var newAvgVals []interface{}

	for i := range example.FeatureSlice {
		switch example.FeatureSlice[i].(type) {
		case float64:
			newAvgVals = append(newAvgVals, 0.0)
		case bool:
			newAvgVals = append(newAvgVals, false)
		case string:
			newAvgVals = append(newAvgVals, "")
		}
	}

	return newAvgVals
}

func findLeast(c1, c2, c3 float64) (int, float64) {
	if c1 <= c2 && c1 <= c3 {
		return 0, c1
	} else if c2 <= c1 && c2 <= c3 {
		return 1, c2
	}

	return 2, c3
}

func runningAvg(oldAvgs []interface{}, newVal dataTypes.Data, n int) []interface{} {
	if len(oldAvgs) < len(newVal.FeatureSlice) {
		oldAvgs = initializeAvgs(newVal)
	}

	for i := range newVal.FeatureSlice {
		//reflect the type of the feature slice index handle float, bool and string (don't worry about bool and str yet)
		switch val := oldAvgs[i].(type) {
		case float64:
			temp := float64(val) * float64(n)
			temp += getFloatReflectVal(newVal.FeatureSlice[i])
			oldAvgs[i] = temp / float64(n+1)
		}
	}

	return oldAvgs
}

func findStds(classSam []*dataTypes.Data, class ClassAvg) []interface{} {

	var stdDev []interface{}
	featureLen := len(classSam[0].FeatureSlice)

	for i := 0; i < featureLen; i++ {
		classTotal := 0.0
		for _, sample := range classSam {
			class.stdDev = append(class.stdDev, 0.0)
			//reflect the type of the feature slice index handle float, bool and string (don't worry about bool and str yet)
			fsample := getFloatReflectVal(sample.FeatureSlice[i])
			fclass := getFloatReflectVal(class.averages[i])
			classTotal += math.Pow((fsample - fclass), 2)
		}
		stdDev = append(stdDev, classTotal/float64(class.count))
	}

	return stdDev
}

func findIndex(entropyVals []float64) int {
	minVal := entropyVals[0]
	minIndex := 0

	for i, contender := range entropyVals {
		if contender < minVal {
			minIndex = i
			minVal = contender
		}
	}

	return minIndex
}

func getFloatReflectVal(val interface{}) float64 {
	v := reflect.ValueOf(val)
	v = reflect.Indirect(v)

	floatVal := v.Convert(reflect.TypeOf(0.0))
	return floatVal.Float()
}

func getBoolReflectVal(val interface{}) bool {
	v := reflect.ValueOf(val)
	v = reflect.Indirect(v)

	boolVal := v.Convert(reflect.TypeOf(true))
	return boolVal.Bool()
}

func getStrReflectVal(val interface{}) string {
	v := reflect.ValueOf(val)
	v = reflect.Indirect(v)

	strVal := v.Convert(reflect.TypeOf(""))
	return strVal.String()
}
