 
// Problem Solving - Counting Sort 1 - Easy
        static int[] countingSort(int[] arr) {
        int max = 0;
        for (int i = 0; i < arr.length; i++)
        {
            if(arr[i] > max) max = arr[i];
        }       
        int [] result = new int[max + 1];
        for (int i = 0; i < arr.length; i++)
        {
            result[arr[i]]++;
        }
        return result;
    }

    


// Problem Solving - 2D Array - DS - Easy
        static int hourglassSum(int[][] arr) {
        int max = 0;
        for (int i = 0; i <4; i++)
        {
            for (int j = 0; j<4; j++)
            {
                int sum = arr[i][j] + arr[i][j+1] + arr[i][j+2] + arr[i+1][j+1] + arr[i+2][j] + arr[i+2][j+1] + arr[i+2][j+2];
                if(i == 0 && j == 0) max = sum;
                if(max < sum) max = sum;
            }
        }
        return max;

    }

    

