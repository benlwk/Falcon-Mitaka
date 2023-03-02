#include <stdint.h>
#include "cuda_kernel.cuh"
// #ifndef TEST_VECTOR_CUH__
// #define TEST_VECTOR__

// int8_t f [512]= {1, 4, 4, 2, -4, -4, -2, -2, 6, -2, 0, -4, 5, 4, 4, 2, -3, -2, -3, -11, -2, -2, -2, 3, 1, 3, -3, -4, -1, 2, 3, 2, 2, -2, -5, 0, 1, -3, -8, -1, 1, 1, -2, -1, 2, -1, -3, 0, 4, 4, 4, 3, 3, 4, 0, 4, 3, -6, -7, -2, 4, 3, -3, -2, 4, 0, 2, 5, -1, 0, 4, -3, 5, 1, 3, 8, 3, 2, -4, 0, 2, 1, 1, 6, 7, 2, -3, 8, 2, -2, -6, 0, 5, -3, 5, -1, 1, -1, 5, 1, 6, -6, 9, 0, 4, -4, 6, -3, -2, 3, -9, 4, 1, -3, 5, -2, 1, -4, 4, -7, 5, -1, 3, -7, 1, 4, 1, 5, -1, 4, 2, -3, 3, -1, 1, -3, 1, -2, -4, 0, 0, -3, -1, 1, -4, -4, -1, -3, 7, -4, 0, -1, -5, -4, 2, -8, -1, 6, -7, -5, 2, -5, -10, -3, 3, -2, 2, -2, 5, -3, 5, 3, 1, -2, -1, -2, 3, -5, -1, 6, -2, 1, -2, -1, -3, 3, -5, 1, -2, 7, 1, 1, -11, 2, 2, 3, -5, 3, -9, 2, -3, 4, 1, -5, -2, 4, -5, 4, 5, 0, -1, 4, 0, -1, 0, -9, -4, -1, -5, 0, 4, -2, 3, -2, -8, 2, 6, 0, 2, 2, -2, 3, -2, -2, 1, -5, -1, -2, 1, 3, -3, 0, -5, 6, -1, -5, 6, -1, 8, 0, 0, 7, -1, -5, -1, -6, 3, -1, -3, -17, -2, -4, -6, 3, -6, -1, -6, -2, -1, 0, -1, -3, -4, -1, 4, 3, -1, 6, -4, -1, -4, 5, 0, 10, 6, 7, 6, 3, 2, 0, 1, -3, 0, -1, 8, 0, -7, 2, 4, -5, -4, 4, -1, -3, 5, -3, 0, 0, -4, 0, 5, -6, 5, -1, 6, 0, -8, 4, -1, -1, 1, -3, -5, 2, 9, 4, 1, 8, 5, 8, -6, 7, 1, 5, 0, -2, -8, 6, -2, 6, 2, 0, 9, 3, -2, 1, -1, -1, 0, -5, -3, 0, 0, -1, 1, -2, -4, -3, -8, 2, -2, -1, -1, 0, 1, 2, 3, 1, -3, 0, -2, -3, -4, 7, 1, -2, 1, 3, -1, -11, -1, -3, 0, 0, 1, 3, 1, 0, 3, 4, -3, 9, -3, 2, 1, -4, 5, 2, -2, 0, -5, 3, -8, 1, 2, -1, -3, -4, 4, -1, 1, -1, -2, 5, -3, -1, 5, -2, 1, -4, 5, -1, -4, -13, -1, 4, -5, 3, 0, -1, -2, -4, -1, -2, -5, 4, 6, 3, 3, -1, 2, 3, 6, -3, -3, 5, -4, 5, -4, 6, -1, 5, 0, -5, 3, 0, -2, 4, -7, 0, 4, -2, 4, 0, -7, -2, 5, 4, 5, 5, 3, 2, -2, -2, -6, -4, 3, 0, -2, -2, -1, 6, 2, -4, -1, -2, -4, -4, -2, -3, -3, 3, 0, -2, -2, 0, 2, -1, 10, -2, 2, -1, 0, 1, -4, 1, -3, 3, -3, 1, -1, -2};
// int8_t g [512]= {2, 5, 5, -5, 0, 1, 5, 1, -8, 0, 5, 4, -3, -2, -2, 0, 1, -4, -6, -3, -1, -1, -3, 2, 4, -6, 2, -7, -2, 10, 3, -2, 1, 1, 0, -4, 5, -2, 2, 2, 2, -5, 7, 3, 3, 3, 3, 0, 3, -1, -1, -4, 0, 0, 1, -3, 6, 0, 5, -4, -1, -7, 3, 3, 4, 1, -7, -2, 4, 3, 1, 0, -1, 4, -2, -2, 1, 2, 4, -2, 1, -6, -2, 0, -1, -2, -1, 8, 3, -5, -2, -3, 2, 1, 2, -4, -2, -8, -2, -4, 2, 7, -2, -8, -1, -1, 4, 1, 3, 2, -6, 1, 0, 2, -4, -2, -4, -5, -4, 1, -4, 7, -7, 1, -2, -4, 1, -1, 11, 0, -2, -8, -3, 3, -3, 0, 7, 5, -10, 1, -1, -4, -6, -2, 1, -4, 1, -2, 2, -1, 5, -2, -6, -2, -6, 3, 3, 5, 5, 0, 0, -1, -3, -2, -4, -2, 2, 4, 0, -3, 7, -1, -1, -2, 4, 5, -2, 4, 3, 2, 2, -3, -4, 6, 1, -1, -1, -2, -3, 6, -7, -2, -1, -1, 2, 0, 0, 4, 0, 13, -6, 3, 0, 0, 3, -13, -2, 2, -5, -7, -10, 2, -6, 4, -5, -1, -4, -4, -5, 4, -6, 1, -8, 1, 0, 2, 4, 3, 4, 2, -5, -4, 2, 0, -4, -5, 2, 0, 4, 4, 6, -6, -7, 2, -3, -1, 1, 0, -2, 3, -5, 3, 4, 1, 0, 3, 6, -4, 1, 1, 3, -6, -1, -7, -4, 0, 1, 0, 4, 1, 12, -6, 2, 9, -1, -2, -3, -5, -6, 3, 5, -6, 0, 2, 3, -2, -4, 1, 0, 5, 10, -6, 1, 1, 1, -2, 4, 0, -2, 2, 7, 2, -3, 1, -3, 4, -3, 14, -3, -5, 0, 0, -4, 2, -7, 1, -2, 2, -4, -8, 2, 3, 3, -2, 2, 10, 7, -9, 1, -1, -2, 2, -5, 7, -3, 2, -6, 1, 0, 4, 1, 1, 4, 3, -6, 3, 1, -5, 4, -3, 3, -3, -1, -8, -2, 3, 3, -7, -1, 5, -1, -4, -11, -2, 1, 0, -3, 1, 3, -1, -2, -7, -3, 2, 13, -11, 2, 6, 3, 1, 7, 5, -1, -6, 5, 4, -5, 0, 2, 0, -4, 2, -9, 0, -3, 2, 2, 2, 1, 5, 1, -7, 0, 2, 2, -4, -2, 0, -5, -1, -2, -4, -5, 4, -2, -1, -3, 2, 0, 10, -1, 0, 3, 0, 8, 6, 3, 0, 1, 3, 0, -2, -2, 1, -1, -6, 4, 4, 4, 7, -7, 1, 5, -6, -4, 1, -2, 1, 8, 2, -1, 1, -7, 4, 5, 3, -1, -2, 8, 6, -6, -13, -6, 8, 6, -5, 4, -1, 4, -7, 8, 4, 0, -1, -3, -6, 0, 0, 5, 4, 4, 2, -7, -1, -1, 2, 5, 3, -7, -1, -3, 10, -4, -1, 4, -1, 1, -2, 6, 1, -9, 3, 5, -2, 9, 0, -3, 5, 0, 3, 0, -1}; 
// int8_t F [512]= {45, -35, -49, 30, 21, 19, -29, -17, 21, -24, -36, 19, 9, -27, 10, -18, 3, -17, -36, 23, 8, 23, 6, -3, 3, -26, -20, -28, -13, 14, -67, 25, 9, 5, 25, 6, -23, 12, -24, 6, -21, 11, 25, -25, 25, -17, -5, -15, 13, 13, -15, -36, 12, -10, -15, -12, -8, -2, -5, -23, -7, 85, 14, 33, 7, -4, -36, -53, -33, -23, -12, -9, -18, 26, -8, 20, 33, 21, -7, 16, 0, 42, -14, -11, -1, 20, 26, -38, 34, 10, -20, -2, 4, 12, -17, 11, 41, -21, 32, 25, 48, -14, -45, -28, 1, -27, -34, -17, -12, -35, -22, 23, -15, -2, 20, 18, 23, -8, 28, 54, 5, 1, 9, -8, -10, 31, 2, -35, 25, -7, 3, 16, -57, -12, 2, 8, -23, 5, 44, 57, 66, -8, -1, -14, -52, -7, -3, -8, 60, -6, 18, -36, 9, 28, 13, 2, -16, 4, 17, -11, 40, 30, 64, -41, -7, 45, -70, 17, -41, 61, 4, -63, 11, -3, 19, -26, 23, 17, 10, -13, -19, 5, -10, -49, -25, 5, -32, -9, 14, 31, -8, 5, 51, -4, 18, 12, 0, 44, -24, 31, -11, 38, 56, 25, 15, -29, -2, -42, -16, -5, -69, 35, -26, -12, 8, -17, 50, 34, 11, 19, -35, 39, -16, 7, -27, -6, 0, -41, 38, 20, -16, -29, 2, 33, 7, 7, -20, 17, 30, 7, 14, 42, 3, 45, -7, 29, -29, -4, -24, 0, -15, -7, -14, -9, -2, 23, 1, 1, 24, 4, 18, -53, -47, -23, 0, 25, -14, 1, 21, 34, -38, -22, -19, 19, -8, -27, -12, 37, -36, -17, 36, -32, 28, -26, 20, -25, -36, -20, 1, -14, -12, -7, 20, -12, 1, 1, 7, -19, 38, -30, -23, -33, 11, -11, -16, 7, -22, 7, -6, -5, -58, -41, -26, 7, -6, -4, -3, 39, 13, -3, 13, 23, -4, 78, -16, -18, 0, 7, 26, -20, -34, 9, -8, -14, 21, -31, 19, -8, 2, 9, -52, -13, 8, -41, -26, 37, 30, -50, 14, -33, -19, 12, -55, -12, 5, 11, 39, 20, -10, 27, -9, 3, -16, -21, -15, 4, 1, 13, -21, -5, -14, 26, -4, 27, -16, 24, 35, -2, -34, -6, -9, -8, 7, -29, -13, 2, 10, -21, 1, -2, 25, -18, -24, -23, 13, 0, -27, -6, -19, 30, -3, -10, 40, -27, -16, -26, -16, -4, 19, -12, -5, 5, -5, 11, 9, -22, 10, 14, 8, -18, 19, 41, 50, 18, -23, 12, -28, -2, -14, 35, -12, -1, 3, 11, -21, -19, 27, 64, 46, -46, -10, 23, 17, 2, -68, 12, -7, -23, -13, 53, -19, 12, 1, -6, -16, -2, -6, -28, 29, -16, 5, 10, 22, 44, 17, 23, 28, -39, 11, -18, 33, 18, 24, -19, -6, -6, 15, 3, -12, 23, 20, 18, -13, 25, -42, 11, 1, -6, -18, 31, 40, 35, -16, -42, -17, 18, -42, -33, -22, -5, -4, 23, 13, -20, -38, 6, -25, -50, -43, 0, 3, 30};
// int8_t G [512]= {16, -19, -45, -4, 26, 13, -36, -49, -8, -8, 1, -3, -22, -17, -55, -11, -88, 0, 33, 55, -11, 3, 24, 15, -19, -10, 58, 69, 20, -50, -41, -31, 5, -30, 10, -30, -9, 29, -7, 0, -13, 23, 7, -16, 12, 35, 2, 30, 43, 18, -27, -3, -43, -42, 11, -58, -48, 19, 35, -3, 6, -14, 34, -46, -33, -51, 19, 68, 18, -23, -34, 25, -24, -33, 30, 51, 18, 12, 7, 3, -8, 6, 13, -17, -12, -11, 23, -11, 59, 29, 31, 18, -48, -20, -10, 8, 29, -7, -8, 12, -26, 28, -28, 21, -11, -10, -11, -10, 16, -27, 1, 21, -13, -63, -4, -4, 5, 63, 25, -12, 15, -8, 9, -18, -16, 46, -9, 10, -17, -1, -4, -12, -20, -5, 1, -18, -11, -4, 17, -18, 65, -1, 1, -18, -25, 12, -4, 4, 25, -30, 20, -21, 9, 9, -48, 9, 21, 51, 23, -6, -27, -22, -35, -32, -22, 0, -33, 32, -7, -4, -11, -43, 35, 19, 28, 2, -13, 17, 16, 9, -7, 14, -46, -23, -17, -28, 1, 6, 24, 27, 37, 18, -43, 39, -17, 24, 9, 24, -17, -18, -24, 18, -23, -34, 0, 16, 44, -6, -11, -16, 41, 17, -29, -15, 16, 22, 10, 9, -1, 2, 58, -4, -11, -3, 13, -6, -32, 14, -22, 15, 28, -23, -19, -6, -20, 1, -27, -32, 1, -26, 33, -33, 6, -43, -12, 67, 10, 64, -16, 7, 53, 36, 4, 16, -44, 25, -12, 25, -15, -5, -16, -64, -12, -41, 29, -16, 0, -14, 7, -17, -28, 38, 71, 43, -6, -39, -14, 12, -25, -11, -30, 37, -74, -36, 31, 22, 14, 6, 8, 38, -23, -33, -27, -10, 24, -34, 45, -29, -5, 8, -20, 33, -6, -24, -3, -13, 11, -17, -20, 4, -46, 7, -28, -28, 29, 5, 59, -20, 34, -16, 28, 35, -28, -61, 32, -3, 52, -46, 8, 18, 30, -7, -56, -18, 5, 19, -22, 45, 56, -2, -10, 40, 8, -46, 1, 84, 13, -45, -5, -31, 26, -39, 15, -17, 28, 8, -64, -24, -18, 49, -55, -42, -13, 39, -12, 42, -69, 24, 12, 31, 3, -18, 13, -6, 7, 1, 9, 82, 7, -14, 4, -8, -20, -17, 24, 5, -20, 6, -1, 47, -29, -40, -51, 80, 18, 30, -17, 8, 57, 21, -26, -31, 22, 21, -11, -72, -5, 4, 30, -28, -12, 29, -15, -14, -22, 3, 46, 20, 65, 19, -4, 32, 14, -9, -42, 44, -6, -22, -12, -36, 23, 24, -25, -16, 5, -13, 6, -6, -29, -16, 2, 11, -4, -36, 18, 24, -8, 10, -34, -28, 41, 38, 9, 0, -17, 14, -38, -28, -8, 31, -27, -28, -31, -57, 22, -21, 22, 40, -32, 45, -19, 0, -26, -23, 12, 17, 2, 51, -18, -7, -15, 9, -32, -29, 3, 27, 8, 19, -7, 9, -15, -22, -22, 64, 60, -24, -18, 4, 41, 21, -9, -49, -1, 2, -4, -54, -21, 12, 50, -57, 28, 22};
int16_t hm[512]= {5856, 9672, 354, 9719, 7445, 6283, 10462, 8588, 85, 3840, 154, 4853, 9810, 2498, 6289, 11201, 10788, 1348, 4887, 9929, 6350, 11349, 9022, 8339, 1628, 6696, 7021, 7449, 11889, 6671, 6476, 8912, 5472, 9103, 11621, 4985, 11799, 7779, 6236, 4880, 3469, 2493, 455, 8968, 7869, 9871, 3795, 10502, 24, 9231, 1668, 10999, 5937, 10579, 6780, 1407, 149, 922, 884, 5092, 7485, 10910, 3231, 9806, 9181, 4821, 7829, 10197, 9857, 1682, 5559, 9831, 10176, 3765, 10055, 12080, 5137, 11767, 7822, 2386, 7572, 7172, 2543, 3735, 5842, 9635, 6402, 11416, 8387, 9006, 9905, 6464, 7783, 8087, 469, 5723, 8863, 1320, 4171, 4942, 8036, 7169, 1063, 1414, 4290, 4015, 2886, 5520, 7406, 6283, 1727, 757, 2589, 5012, 10398, 347, 2945, 4993, 3535, 12102, 4117, 10037, 9469, 3593, 607, 6667, 11300, 2911, 5494, 10477, 4953, 5479, 8012, 6745, 4327, 7739, 4946, 1830, 8740, 11249, 10418, 1337, 2159, 6103, 10419, 6852, 10918, 634, 2147, 11027, 115, 11934, 2008, 5283, 9696, 5339, 8111, 10592, 4456, 3920, 11139, 7080, 6830, 275, 11516, 7928, 8530, 7695, 1683, 11501, 5871, 11165, 7100, 810, 7857, 5717, 10166, 9130, 11194, 10671, 536, 737, 12248, 6353, 8544, 3937, 5034, 3917, 8711, 8462, 10378, 3165, 4493, 9024, 2140, 9035, 7460, 4719, 3552, 7015, 4469, 9174, 5994, 9150, 6422, 11130, 3000, 1976, 8802, 2033, 11441, 1864, 8543, 611, 2935, 11438, 2307, 5860, 5554, 741, 6519, 8885, 2954, 5716, 6994, 3536, 7164, 5280, 3461, 6095, 5498, 11397, 11592, 9323, 12119, 5137, 2923, 5547, 9278, 6730, 2513, 8428, 10364, 2322, 12010, 11719, 4476, 8565, 1958, 2343, 2126, 6759, 6207, 8633, 9496, 5827, 5339, 2665, 4457, 12239, 6369, 1174, 6305, 10726, 6062, 1470, 40, 3572, 5825, 9256, 3030, 8189, 4495, 10957, 7556, 10444, 3929, 3336, 7719, 7701, 3920, 3240, 5384, 6158, 10377, 9603, 11575, 11387, 6515, 383, 10946, 10338, 10925, 9516, 2561, 9248, 11489, 5989, 5931, 10860, 5884, 9448, 12029, 4249, 4721, 11411, 6516, 919, 9414, 4021, 2813, 6433, 4060, 6229, 9324, 8396, 3655, 4038, 11401, 11391, 6624, 6965, 1187, 7778, 9525, 5290, 4589, 1770, 9836, 1847, 5239, 11924, 4889, 8212, 1707, 5735, 4974, 2531, 449, 2768, 12064, 9859, 10323, 3459, 4735, 9661, 384, 1189, 7548, 5661, 11844, 9595, 6749, 1936, 6998, 11774, 2877, 3385, 2207, 9872, 4350, 11077, 9953, 455, 2552, 8605, 9946, 6442, 7246, 3157, 10849, 4656, 1667, 9455, 1555, 8596, 5495, 888, 8049, 11028, 3015, 10324, 2961, 2816, 9379, 8914, 5548, 6966, 8344, 8972, 8966, 224, 6524, 7962, 7014, 4269, 11969, 9655, 6309, 11034, 2971, 417, 2194, 9184, 8810, 2677, 763, 7385, 8099, 1985, 11618, 4343, 4646, 4330, 2255, 3333, 4466, 6455, 7587, 7483, 6205, 4553, 5597, 3179, 6977, 6462, 6934, 7464, 11732, 9158, 7192, 4401, 6097, 11916, 640, 9047, 1129, 6235, 7792, 10154, 7615, 11712, 4726, 6950, 4287, 5438, 3839, 3522, 8654, 287, 11696, 6806, 2209, 8790, 8208, 9387, 7654, 10043, 11137, 3879, 11513, 11394, 335, 3681, 5862, 5871, 3232, 91, 6080, 3237, 4310, 6144, 12168, 10215, 8661, 10679, 10909, 2373, 7161, 7496, 11050, 6570, 555, 11779, 1676, 3613, 8205, 7877, 2609, 3911, 11342, 570, 8790, 11693, 6020, 2933, 5761, 7284, 1002, 8389, 8784, 2452, 5762, 3906, 7918, 4778, 461, 7072, 5622, 7193, 11329, 3073};
uint8_t seed_tv[48] = {196, 115, 28, 56, 25, 242, 155, 120, 71, 77, 165, 234, 163, 139, 94, 152, 178, 83, 84, 125, 118, 108, 242, 61, 80, 232, 135, 49, 192, 28, 166, 225, 24, 108, 247, 80, 78, 54, 174, 18, 180, 233, 232, 20, 192, 110, 104, 235};
uint8_t nonce_tv[NONCELEN] = {51, 179, 192, 117, 7, 228, 32, 23, 72, 73, 77, 131, 43, 110, 226, 166, 201, 59, 255, 155, 14, 227, 67, 181, 80, 209, 248, 90, 61, 13, 224, 215, 4, 198, 209, 120, 66, 149, 19, 9};
// uint8_t m_tv[MLEN] = {216, 28, 77, 141, 115, 79, 203, 251, 234, 222, 61, 63, 138, 3, 159, 170, 42, 44, 153, 87, 232, 53, 173, 85, 178, 46, 117, 191, 87, 187, 85, 106, 200};
// __device__ static uint8_t prng_tv [512] = {186, 79, 105, 238, 10, 121, 221, 116, 129, 144, 135, 164, 133, 197, 217, 152, 103, 35, 158, 127, 28, 95, 248, 109, 226, 214, 164, 91, 214, 130, 61, 99, 157, 24, 24, 36, 66, 177, 199, 251, 36, 250, 231, 98, 129, 169, 216, 192, 76, 149, 220, 191, 161, 221, 125, 233, 190, 61, 41, 109, 109, 248, 138, 126, 235, 112, 20, 244, 146, 32, 220, 103, 12, 109, 218, 72, 173, 194, 203, 251, 3, 221, 29, 83, 207, 232, 161, 224, 35, 251, 199, 238, 155, 129, 235, 137, 28, 253, 109, 83, 180, 59, 219, 162, 56, 70, 248, 85, 25, 238, 213, 149, 136, 76, 102, 116, 185, 161, 194, 94, 3, 11, 182, 182, 7, 216, 114, 113, 39, 148, 92, 191, 18, 59, 32, 101, 62, 32, 241, 188, 134, 76, 192, 33, 74, 36, 100, 179, 144, 27, 158, 103, 111, 74, 13, 200, 215, 160, 63, 60, 220, 25, 24, 134, 104, 117, 201, 49, 70, 175, 234, 78, 162, 149, 123, 227, 158, 239, 241, 51, 153, 232, 65, 215, 252, 70, 88, 172, 68, 68, 98, 236, 129, 167, 111, 109, 223, 234, 42, 213, 11, 17, 254, 52, 34, 176, 27, 155, 238, 213, 32, 49, 111, 58, 130, 96, 192, 206, 228, 65, 63, 84, 185, 87, 98, 206, 118, 10, 205, 30, 37, 74, 44, 87, 130, 13, 112, 184, 158, 7, 77, 205, 110, 233, 35, 190, 11, 153, 248, 1, 185, 144, 114, 110, 18, 221, 76, 124, 227, 211, 5, 8, 105, 49, 153, 244, 60, 55, 63, 57, 164, 182, 220, 100, 118, 11, 136, 140, 119, 17, 31, 9, 143, 208, 61, 51, 118, 106, 91, 147, 179, 207, 179, 126, 195, 54, 47, 208, 155, 33, 159, 102, 94, 104, 216, 153, 144, 247, 112, 42, 212, 182, 5, 40, 60, 150, 150, 38, 109, 7, 80, 213, 30, 125, 242, 222, 153, 153, 11, 110, 220, 95, 28, 104, 57, 109, 26, 220, 252, 151, 181, 162, 78, 8, 139, 70, 87, 105, 163, 193, 139, 137, 169, 108, 64, 1, 174, 81, 3, 140, 247, 157, 105, 214, 134, 12, 82, 108, 46, 106, 78, 42, 69, 103, 165, 127, 99, 192, 102, 181, 101, 224, 204, 74, 109, 45, 163, 59, 120, 145, 112, 248, 108, 139, 164, 186, 54, 1, 246, 114, 7, 32, 58, 122, 36, 143, 124, 29, 163, 164, 218, 16, 24, 103, 7, 195, 23, 233, 199, 207, 205, 52, 226, 147, 162, 178, 84, 223, 242, 111, 19, 125, 54, 144, 82, 76, 185, 155, 220, 60, 57, 88, 138, 66, 102, 111, 239, 39, 129, 170, 45, 5, 90, 147, 242, 223, 29, 123, 228, 110, 248, 33, 70, 243, 232, 248, 225, 119, 184, 120, 177, 74, 153, 250, 77, 110, 234, 226, 92, 79, 114, 158, 181, 16, 231, 242, 217, 27, 218, 181, 231, 87, 177, 125, 1, 173, 157, 210, 188, 220, 178, 54, 181, 59, 90, 70, 233, 103, 173, 21, 172, 218};
// uint64_t rng_A[25] = {8690806096070996932, 10979366477708479815, 4463749436842202034, 16259715116458043472, 1346073048624098328, 16962929767754557876, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9223372036854775808, 0, 0, 0, 0, 0, 0, 0, 0};
// uint8_t rng_dbuf[200] = {196, 115, 28, 56, 25, 242, 155, 120, 71, 77, 165, 234, 163, 139, 94, 152, 178, 83, 84, 125, 118, 108, 242, 61, 80, 232, 135, 49, 192, 28, 166, 225, 24, 108, 247, 80, 78, 54, 174, 18, 180, 233, 232, 20, 192, 110, 104, 235, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
// uint16_t c0[512]= {5856, 9672, 354, 9719, 7445, 6283, 10462, 8588, 85, 3840, 154, 4853, 9810, 2498, 6289, 11201, 10788, 1348, 4887, 9929, 6350, 11349, 9022, 8339, 1628, 6696, 7021, 7449, 11889, 6671, 6476, 8912, 5472, 9103, 11621, 4985, 11799, 7779, 6236, 4880, 3469, 2493, 455, 8968, 7869, 9871, 3795, 10502, 24, 9231, 1668, 10999, 5937, 10579, 6780, 1407, 149, 922, 884, 5092, 7485, 10910, 3231, 9806, 9181, 4821, 7829, 10197, 9857, 1682, 5559, 9831, 10176, 3765, 10055, 12080, 5137, 11767, 7822, 2386, 7572, 7172, 2543, 3735, 5842, 9635, 6402, 11416, 8387, 9006, 9905, 6464, 7783, 8087, 469, 5723, 8863, 1320, 4171, 4942, 8036, 7169, 1063, 1414, 4290, 4015, 2886, 5520, 7406, 6283, 1727, 757, 2589, 5012, 10398, 347, 2945, 4993, 3535, 12102, 4117, 10037, 9469, 3593, 607, 6667, 11300, 2911, 5494, 10477, 4953, 5479, 8012, 6745, 4327, 7739, 4946, 1830, 8740, 11249, 10418, 1337, 2159, 6103, 10419, 6852, 10918, 634, 2147, 11027, 115, 11934, 2008, 5283, 9696, 5339, 8111, 10592, 4456, 3920, 11139, 7080, 6830, 275, 11516, 7928, 8530, 7695, 1683, 11501, 5871, 11165, 7100, 810, 7857, 5717, 10166, 9130, 11194, 10671, 536, 737, 12248, 6353, 8544, 3937, 5034, 3917, 8711, 8462, 10378, 3165, 4493, 9024, 2140, 9035, 7460, 4719, 3552, 7015, 4469, 9174, 5994, 9150, 6422, 11130, 3000, 1976, 8802, 2033, 11441, 1864, 8543, 611, 2935, 11438, 2307, 5860, 5554, 741, 6519, 8885, 2954, 5716, 6994, 3536, 7164, 5280, 3461, 6095, 5498, 11397, 11592, 9323, 12119, 5137, 2923, 5547, 9278, 6730, 2513, 8428, 10364, 2322, 12010, 11719, 4476, 8565, 1958, 2343, 2126, 6759, 6207, 8633, 9496, 5827, 5339, 2665, 4457, 12239, 6369, 1174, 6305, 10726, 6062, 1470, 40, 3572, 5825, 9256, 3030, 8189, 4495, 10957, 7556, 10444, 3929, 3336, 7719, 7701, 3920, 3240, 5384, 6158, 10377, 9603, 11575, 11387, 6515, 383, 10946, 10338, 10925, 9516, 2561, 9248, 11489, 5989, 5931, 10860, 5884, 9448, 12029, 4249, 4721, 11411, 6516, 919, 9414, 4021, 2813, 6433, 4060, 6229, 9324, 8396, 3655, 4038, 11401, 11391, 6624, 6965, 1187, 7778, 9525, 5290, 4589, 1770, 9836, 1847, 5239, 11924, 4889, 8212, 1707, 5735, 4974, 2531, 449, 2768, 12064, 9859, 10323, 3459, 4735, 9661, 384, 1189, 7548, 5661, 11844, 9595, 6749, 1936, 6998, 11774, 2877, 3385, 2207, 9872, 4350, 11077, 9953, 455, 2552, 8605, 9946, 6442, 7246, 3157, 10849, 4656, 1667, 9455, 1555, 8596, 5495, 888, 8049, 11028, 3015, 10324, 2961, 2816, 9379, 8914, 5548, 6966, 8344, 8972, 8966, 224, 6524, 7962, 7014, 4269, 11969, 9655, 6309, 11034, 2971, 417, 2194, 9184, 8810, 2677, 763, 7385, 8099, 1985, 11618, 4343, 4646, 4330, 2255, 3333, 4466, 6455, 7587, 7483, 6205, 4553, 5597, 3179, 6977, 6462, 6934, 7464, 11732, 9158, 7192, 4401, 6097, 11916, 640, 9047, 1129, 6235, 7792, 10154, 7615, 11712, 4726, 6950, 4287, 5438, 3839, 3522, 8654, 287, 11696, 6806, 2209, 8790, 8208, 9387, 7654, 10043, 11137, 3879, 11513, 11394, 335, 3681, 5862, 5871, 3232, 91, 6080, 3237, 4310, 6144, 12168, 10215, 8661, 10679, 10909, 2373, 7161, 7496, 11050, 6570, 555, 11779, 1676, 3613, 8205, 7877, 2609, 3911, 11342, 570, 8790, 11693, 6020, 2933, 5761, 7284, 1002, 8389, 8784, 2452, 5762, 3906, 7918, 4778, 461, 7072, 5622, 7193, 11329, 3073};
// int16_t s2[512]= {135, -406, 143, 158, 280, -34, -233, 39, -165, -357, 129, -189, 86, -350, -118, 4, -40, -140, 5, -96, 94, 41, -51, -89, -311, 107, 1, -213, -228, -25, 315, -212, -106, 215, 61, -19, 112, -53, -8, 50, -181, 135, 102, -89, -193, 115, -59, 27, 81, 57, -260, -123, 76, -255, 190, 73, 18, -262, -216, -77, 272, -180, 101, -30, 177, 40, 212, -110, -448, 248, 23, 15, 13, -171, -261, 19, -124, 97, -136, 364, -82, 74, 339, -114, -308, -68, -2, -234, 166, 82, 211, -135, 76, -95, -190, 395, -15, -249, 87, 29, -81, 204, -231, -181, 234, -145, -70, -222, 238, 356, 10, 50, 48, 222, 123, 83, -97, 269, 88, -12, 124, -113, -335, 73, -225, -251, -12, -89, 45, 167, -72, 212, 10, -117, 147, 109, -252, -289, -54, -274, -81, -159, 1, 77, 244, -95, -124, 70, -202, 35, -166, -9, -60, 157, 38, -44, -109, -19, -137, 37, 80, -3, 183, 176, 70, 24, -108, 170, -241, -72, 151, 2, -26, 257, 20, -66, -27, 251, 5, -1, 13, 355, 178, 69, -181, -71, 32, 198, 128, -215, 268, -196, 44, 132, -28, -86, 33, -153, 98, 460, 70, -4, 34, 14, -86, 181, -178, 35, -676, -86, -302, -177, 200, 85, 381, 125, -64, 95, -28, 22, 71, -70, -289, -76, -132, -261, -50, -105, 101, 75, 150, -238, 6, 131, -10, -3, -347, 48, 334, -105, 32, 25, -239, 70, -160, 52, 125, 59, 94, 217, -254, -62, -146, 151, 2, -188, -176, 348, -174, 71, 74, -64, 281, -3, -222, 246, 95, -67, -36, -202, 238, -185, 117, -146, -73, 177, -84, -155, 252, -179, -7, 131, -24, -175, 213, -45, 39, -87, -50, -231, 224, 108, 113, 112, -131, 118, -31, 44, 171, 73, 13, -44, 79, 276, 172, 207, 197, 252, 215, -192, -151, 68, -238, 205, -183, -55, -108, -76, 119, -14, -250, 158, 222, 149, 154, 100, -184, 272, -20, 144, 79, -174, 113, 120, -338, 107, -82, -167, -55, -69, 289, 151, -37, 32, 125, -335, 18, -18, 33, 208, -161, 446, -172, 303, -47, 221, 37, -177, 102, 51, -120, 62, -79, -16, 118, -236, -64, 0, -87, -102, -112, -25, 278, 196, -244, -3, -270, -311, 224, 8, 12, 112, -103, -226, -143, 342, -52, 160, 143, -108, -140, 121, 194, -446, -1, 137, 139, -61, 283, -155, -106, -63, -32, -187, -23, 206, -45, 102, -201, -379, 15, 12, 42, 142, 164, -101, -140, -213, -117, -27, -5, -184, -48, -137, -149, 9, -138, 58, -92, 141, 213, -46, 153, 43, 84, -44, -61, 168, 91, -200, -101, -38, 204, 90, -291, -200, 317, -252, 54, 270, 224, 79, 122, -243, 362, 381, 92, 113, 24, 232, 227, -262, -145, 2, -195, -76, -142, 37, 129, 8, -34, 232, -15, 68, -3, -100, 184, -73, -94, -146, 33, -78, -281, 234, -151, -105, -269, 221, -145, 88, -435, -45, -153, -167, -189, -52, 144, 100, 48, -108, -28, -215, -94, -131, 302, 132, 21, -80, 152, -85, -353, 167};
uint16_t h[512]= {2967, 6893, 10256, 4032, 12035, 8280, 6629, 10074, 10662, 3179, 2548, 250, 10744, 4943, 8274, 4605, 1556, 5076, 6013, 2414, 1175, 11900, 11611, 1342, 5659, 12077, 10011, 5458, 6410, 528, 11141, 1139, 1834, 11085, 441, 1476, 3483, 6491, 747, 2941, 1854, 4678, 5157, 4794, 6216, 9959, 2495, 4788, 9054, 3799, 5594, 1883, 8026, 350, 9629, 8176, 3034, 11179, 7720, 10513, 2638, 5569, 3127, 4195, 5772, 6649, 2234, 5161, 11807, 4409, 9474, 5554, 170, 7313, 3645, 1281, 7324, 6331, 5233, 4724, 8973, 2539, 4593, 426, 11374, 12214, 12274, 3386, 721, 2333, 3656, 10625, 1041, 8708, 7400, 2280, 6609, 11383, 5487, 6656, 12248, 10452, 1463, 2604, 4950, 653, 3196, 8839, 3938, 8479, 1736, 11523, 6342, 3381, 943, 1501, 2152, 6689, 7196, 9281, 4391, 4174, 196, 3215, 1341, 9374, 6136, 8428, 7947, 1632, 5718, 9528, 11408, 9153, 7192, 1884, 10690, 10990, 6146, 583, 1925, 3111, 2478, 5396, 7989, 6740, 10921, 1752, 8601, 3347, 9717, 8937, 6227, 7484, 6680, 8117, 4053, 45, 3143, 10998, 6867, 6112, 8405, 1006, 9481, 7195, 2267, 4812, 6575, 9177, 3910, 10305, 10748, 8720, 8236, 867, 455, 10395, 1188, 1065, 3337, 5412, 847, 4862, 7337, 9147, 8856, 8625, 1409, 3840, 11192, 11600, 4472, 9409, 451, 7279, 786, 8219, 936, 4218, 199, 4789, 11959, 8775, 7384, 4030, 5925, 5926, 3964, 3149, 9546, 5655, 6028, 6327, 5215, 4747, 10448, 11286, 11192, 6185, 11843, 7824, 6081, 7278, 11963, 487, 5534, 3135, 11514, 2491, 2902, 9058, 2329, 5926, 3650, 2551, 6615, 4799, 2959, 8134, 8502, 2566, 1024, 8403, 3775, 8565, 6468, 9185, 1197, 7291, 3101, 10802, 3665, 4917, 5667, 3790, 3846, 2904, 9056, 6724, 1509, 10591, 5035, 10040, 2634, 1410, 11544, 2733, 6564, 8464, 4582, 3991, 6145, 995, 11341, 11541, 5571, 39, 12104, 10959, 10165, 996, 6311, 11612, 3912, 6643, 10314, 3219, 6650, 449, 8337, 2935, 11015, 6853, 8876, 1470, 3008, 9482, 2791, 4501, 4287, 4757, 7532, 4360, 11222, 6878, 9033, 4366, 5056, 4157, 5269, 2937, 11842, 1468, 11676, 1032, 7925, 11684, 2267, 109, 10761, 8019, 2328, 10187, 2942, 9985, 12280, 4617, 11676, 880, 11114, 12142, 7826, 252, 7291, 11915, 1529, 6695, 7703, 10188, 6458, 9121, 8720, 9981, 5562, 7745, 4134, 6072, 4365, 5649, 10341, 4513, 8093, 11346, 1849, 4513, 7904, 7219, 12273, 6405, 10262, 5279, 2312, 2935, 4843, 10084, 8699, 8651, 7721, 8890, 3859, 4204, 10541, 3072, 7596, 11091, 11248, 4748, 7272, 5607, 9993, 6222, 4276, 1591, 2112, 7602, 7576, 9645, 2010, 1681, 9620, 5535, 6170, 7075, 6296, 6948, 10048, 9569, 10085, 11624, 10058, 4357, 237, 5792, 6841, 7610, 10192, 1279, 4943, 6441, 10117, 2854, 2099, 9324, 4734, 7723, 9617, 832, 11601, 8267, 9281, 158, 10935, 8008, 857, 4420, 1191, 5204, 8792, 10045, 11675, 10917, 10026, 2313, 716, 8521, 10349, 10961, 6561, 1579, 10776, 6208, 6725, 9222, 352, 2331, 11874, 10737, 10727, 7046, 2155, 6644, 1689, 21, 9687, 9740, 5677, 8295, 7213, 5354, 8128, 835, 1058, 9152, 5382, 11395, 4750, 9347, 10055, 6816, 530, 5639, 5589, 6972, 936, 6596, 2779, 11977, 11215, 149, 7769, 786, 4048, 7748, 11088, 7297, 306, 7285, 8829, 545, 4828, 1068, 8585, 10805, 7151, 7476, 10481, 8077, 8057, 5454, 8785, 5380, 8420, 3192, 6268, 1507, 2403, 11955, 10041, 4118, 6005, 3728};
uint8_t pk[CRYPTO_PUBLICKEYBYTES] = {9, 107, 168, 108, 182, 88, 168, 244, 69, 201, 165, 228, 194, 131, 116, 190, 200, 121, 200, 101, 95, 104, 82, 105, 35, 36, 9, 24, 7, 77, 1, 71, 192, 49, 98, 228, 164, 146, 0, 100, 140, 101, 40, 3, 198, 253, 117, 9, 174, 154, 167, 153, 214, 49, 13, 11, 212, 39, 36, 224, 99, 89, 32, 24, 98, 7, 0, 7, 103, 202, 90, 133, 70, 177, 117, 83, 8, 195, 4, 184, 79, 201, 59, 6, 158, 38, 89, 133, 179, 152, 214, 184, 52, 105, 130, 135, 255, 130, 154, 168, 32, 241, 122, 127, 66, 38, 171, 33, 246, 1, 235, 215, 23, 82, 38, 186, 178, 86, 216, 136, 143, 0, 144, 50, 86, 109, 99, 131, 214, 132, 87, 234, 21, 90, 148, 48, 24, 112, 213, 137, 198, 120, 237, 48, 66, 89, 233, 211, 123, 25, 59, 194, 167, 204, 188, 190, 197, 29, 105, 21, 140, 68, 7, 58, 236, 151, 146, 99, 2, 83, 49, 139, 201, 84, 219, 245, 13, 21, 2, 130, 144, 220, 45, 48, 156, 123, 123, 2, 166, 130, 55, 68, 212, 99, 218, 23, 116, 149, 149, 203, 119, 230, 209, 109, 32, 209, 180, 195, 170, 216, 157, 50, 14, 190, 90, 103, 43, 185, 109, 108, 213, 193, 239, 236, 139, 129, 18, 0, 203, 176, 98, 228, 115, 53, 37, 64, 237, 222, 248, 175, 148, 153, 248, 205, 209, 220, 124, 104, 115, 240, 199, 166, 188, 183, 9, 117, 96, 39, 31, 148, 104, 73, 183, 243, 115, 100, 11, 182, 156, 169, 181, 24, 170, 56, 10, 110, 176, 167, 39, 94, 232, 78, 156, 34, 26, 237, 136, 245, 191, 186, 244, 58, 62, 222, 142, 106, 164, 37, 88, 16, 79, 175, 128, 14, 1, 132, 65, 147, 3, 118, 198, 246, 231, 81, 86, 153, 113, 244, 122, 219, 202, 92, 160, 12, 128, 25, 136, 243, 23, 161, 135, 34, 162, 146, 152, 146, 94, 161, 84, 219, 201, 2, 78, 18, 5, 36, 162, 212, 29, 192, 241, 143, 216, 217, 9, 246, 197, 9, 119, 64, 78, 32, 23, 103, 7, 139, 169, 161, 249, 228, 10, 139, 43, 169, 192, 27, 125, 163, 160, 183, 58, 76, 42, 107, 79, 81, 139, 190, 227, 69, 93, 10, 242, 32, 77, 220, 3, 28, 128, 92, 114, 204, 182, 71, 148, 11, 30, 103, 148, 216, 89, 170, 235, 206, 160, 222, 181, 129, 214, 27, 146, 72, 189, 150, 151, 181, 203, 151, 74, 129, 118, 232, 249, 16, 70, 156, 174, 10, 180, 237, 146, 210, 174, 233, 247, 235, 80, 41, 109, 175, 128, 87, 71, 99, 5, 193, 24, 157, 29, 152, 64, 160, 148, 79, 4, 71, 251, 129, 229, 17, 66, 14, 103, 137, 27, 152, 250, 108, 37, 112, 52, 213, 160, 99, 67, 125, 55, 145, 119, 206, 141, 63, 166, 234, 241, 46, 45, 187, 126, 184, 228, 152, 72, 22, 18, 177, 146, 150, 23, 218, 95, 180, 94, 76, 223, 137, 57, 39, 216, 186, 132, 42, 168, 97, 217, 197, 4, 113, 198, 208, 198, 223, 126, 43, 178, 100, 101, 160, 235, 106, 58, 112, 157, 231, 146, 170, 250, 175, 146, 42, 169, 93, 213, 146, 11, 114, 180, 184, 133, 108, 110, 99, 40, 96, 177, 15, 92, 192, 132, 80, 0, 54, 113, 175, 56, 137, 97, 135, 43, 70, 100, 0, 173, 184, 21, 186, 129, 234, 121, 73, 69, 209, 154, 16, 6, 34, 166, 202, 13, 65, 196, 234, 98, 12, 33, 220, 18, 81, 25, 227, 114, 65, 143, 4, 64, 45, 159, 167, 24, 15, 123, 200, 154, 250, 84, 248, 8, 34, 68, 164, 47, 70, 229, 181, 171, 206, 135, 181, 10, 125, 111, 235, 232, 215, 187, 186, 201, 38, 87, 203, 218, 29, 183, 194, 85, 114, 164, 193, 208, 186, 234, 48, 68, 122, 134, 90, 43, 16, 54, 184, 128, 3, 126, 47, 77, 38, 212, 83, 233, 233, 19, 37, 151, 121, 233, 22, 155, 40, 166, 46, 184, 9, 165, 199, 68, 224, 78, 38, 14, 31, 43, 189, 168, 116, 241, 172, 103, 72, 57, 221, 180, 123, 49, 72, 197, 148, 109, 224, 24, 1, 72, 183, 151, 61, 99, 197, 129, 147, 177, 124, 208, 93, 22, 232, 12, 215, 146, 140, 42, 51, 131, 99, 162, 58, 129, 192, 96, 140, 135, 80, 85, 137, 185, 218, 28, 97, 126, 123, 112, 120, 107, 103, 84, 251, 179, 10, 88, 22, 129, 11, 158, 18, 108, 252, 197, 170, 73, 50, 110, 157, 132, 41, 115, 135, 75, 99, 89, 181, 219, 117, 97, 11, 166, 138, 152, 199, 181, 232, 63, 18, 90, 130, 82, 46, 19, 184, 63, 184, 248, 100, 226, 169, 123, 115, 181, 213, 68, 167, 65, 91, 101, 4, 161, 57, 57, 234, 177, 89, 93, 100, 250, 244, 31, 171, 37, 168, 100, 165, 116, 222, 82, 68, 5, 232, 120, 51, 152, 119, 136, 109, 47, 192, 127, 160, 49, 21, 8, 37, 36, 19, 237, 250, 17, 88, 70, 102, 103, 175, 247, 131, 134, 218, 247, 203, 76, 155, 133, 9, 146, 249, 110, 32, 82, 83, 48, 89, 154, 182, 1, 212, 84, 104, 142, 41, 76, 140, 62};
uint8_t sm[MLEN + CRYPTO_BYTES] = {2, 104, 51, 179, 192, 117, 7, 228, 32, 23, 72, 73, 77, 131, 43, 110, 226, 166, 201, 59, 255, 155, 14, 227, 67, 181, 80, 209, 248, 90, 61, 13, 224, 215, 4, 198, 209, 120, 66, 149, 19, 9, 216, 28, 77, 141, 115, 79, 203, 251, 234, 222, 61, 63, 138, 3, 159, 170, 42, 44, 153, 87, 232, 53, 173, 85, 178, 46, 117, 191, 87, 187, 85, 106, 200, 41, 7, 101, 132, 61, 30, 70, 13, 23, 165, 39, 210, 188, 164, 5, 189, 85, 187, 199, 218, 9, 168, 198, 32, 190, 10, 244, 167, 103, 217, 219, 150, 184, 15, 85, 228, 102, 103, 103, 81, 234, 171, 167, 185, 59, 134, 215, 17, 50, 218, 160, 235, 55, 103, 130, 185, 238, 227, 117, 25, 206, 16, 253, 211, 63, 233, 242, 147, 18, 195, 29, 135, 54, 32, 109, 22, 92, 244, 197, 40, 170, 61, 220, 1, 120, 69, 225, 240, 221, 91, 10, 68, 255, 150, 28, 66, 216, 116, 169, 85, 51, 229, 180, 56, 152, 47, 82, 76, 169, 84, 216, 117, 51, 191, 190, 66, 198, 63, 242, 171, 199, 122, 52, 199, 157, 181, 90, 153, 23, 27, 188, 183, 44, 132, 42, 101, 48, 175, 47, 117, 63, 12, 52, 172, 99, 47, 159, 30, 121, 73, 240, 191, 108, 103, 102, 91, 39, 114, 42, 136, 87, 214, 38, 182, 255, 26, 19, 109, 146, 58, 57, 244, 6, 155, 116, 119, 255, 148, 110, 82, 71, 166, 98, 119, 145, 212, 155, 89, 237, 201, 226, 82, 90, 134, 14, 110, 152, 40, 209, 143, 100, 169, 241, 114, 34, 232, 22, 106, 2, 69, 56, 89, 187, 218, 11, 129, 134, 216, 201, 146, 139, 181, 113, 228, 20, 100, 1, 215, 67, 14, 34, 89, 4, 103, 58, 210, 28, 202, 197, 76, 20, 108, 36, 138, 29, 214, 154, 182, 73, 30, 144, 29, 109, 113, 177, 82, 21, 91, 233, 125, 224, 87, 243, 145, 106, 63, 27, 66, 115, 48, 140, 41, 178, 244, 217, 105, 113, 103, 185, 6, 129, 177, 88, 62, 217, 48, 167, 30, 153, 4, 103, 222, 163, 104, 19, 75, 236, 238, 189, 89, 127, 155, 236, 146, 46, 129, 111, 27, 5, 112, 215, 40, 244, 174, 4, 100, 193, 247, 151, 101, 127, 135, 164, 229, 45, 205, 202, 235, 146, 114, 102, 46, 166, 109, 124, 108, 216, 120, 27, 49, 175, 85, 90, 217, 63, 95, 101, 231, 88, 22, 203, 141, 195, 6, 187, 103, 229, 146, 181, 38, 27, 172, 167, 197, 9, 98, 158, 162, 175, 138, 187, 128, 203, 168, 158, 229, 53, 183, 109, 253, 156, 203, 190, 59, 244, 143, 43, 200, 170, 52, 178, 110, 17, 3, 41, 16, 83, 245, 203, 141, 227, 164, 90, 250, 90, 118, 223, 139, 33, 34, 237, 44, 130, 251, 207, 34, 89, 41, 13, 65, 161, 79, 134, 177, 47, 53, 245, 212, 151, 98, 179, 76, 255, 19, 238, 126, 66, 237, 236, 112, 32, 29, 127, 55, 195, 51, 22, 40, 143, 163, 7, 142, 54, 229, 129, 8, 134, 92, 60, 254, 38, 61, 86, 54, 146, 4, 61, 236, 198, 47, 52, 38, 248, 96, 97, 40, 91, 123, 27, 51, 111, 86, 255, 65, 187, 101, 233, 205, 109, 155, 146, 253, 144, 248, 100, 170, 28, 146, 60, 184, 199, 85, 245, 205, 225, 119, 13, 134, 37, 149, 66, 113, 73, 215, 114, 26, 170, 181, 209, 148, 174, 169, 172, 222, 202, 21, 190, 67, 203, 166, 166, 43, 90, 51, 144, 158, 159, 196, 218, 28, 88, 20, 251, 215, 205, 106, 47, 165, 114, 227, 24, 180, 44, 108, 49, 145, 64, 184, 110, 102, 57, 37, 128, 161, 26, 43, 67, 31, 68, 193, 249, 39, 14, 79, 123, 36, 144, 243, 179, 37, 169, 151, 122, 113, 165, 117, 145, 86, 54, 99, 91, 153, 105, 219, 214, 210, 32, 178, 76, 61, 153, 206, 187, 189, 131, 75, 136, 34, 43, 208, 140, 58, 190, 18, 78, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
uint8_t sk[CRYPTO_SECRETKEYBYTES] = {89, 4, 65, 2, 243, 207, 190, 27, 224, 60, 20, 65, 2, 247, 239, 117, 251, 239, 131, 4, 63, 124, 252, 32, 194, 11, 238, 192, 7, 222, 63, 4, 31, 191, 11, 255, 64, 16, 65, 3, 12, 64, 4, 15, 174, 126, 16, 63, 126, 16, 0, 133, 252, 1, 61, 20, 16, 200, 12, 47, 0, 8, 16, 70, 28, 47, 72, 11, 238, 128, 23, 209, 127, 7, 241, 65, 27, 162, 64, 19, 193, 189, 248, 61, 196, 7, 209, 126, 7, 193, 57, 23, 240, 249, 4, 64, 69, 252, 64, 189, 15, 240, 125, 7, 239, 0, 3, 223, 193, 243, 207, 253, 31, 192, 63, 239, 192, 184, 252, 110, 123, 11, 189, 189, 15, 224, 190, 23, 209, 67, 7, 239, 254, 15, 191, 198, 248, 31, 191, 244, 62, 193, 248, 112, 65, 212, 32, 131, 236, 61, 194, 244, 64, 123, 248, 78, 196, 20, 15, 196, 3, 240, 55, 243, 254, 192, 19, 224, 254, 224, 33, 128, 8, 47, 131, 251, 224, 123, 255, 224, 67, 244, 14, 198, 255, 177, 191, 32, 0, 7, 255, 191, 250, 15, 255, 111, 251, 206, 131, 235, 254, 190, 252, 15, 253, 243, 241, 3, 252, 111, 63, 240, 80, 10, 24, 113, 131, 8, 0, 125, 3, 242, 0, 228, 33, 59, 240, 79, 253, 23, 208, 0, 240, 1, 122, 23, 241, 128, 224, 79, 255, 7, 222, 194, 36, 64, 72, 20, 142, 135, 4, 80, 62, 224, 111, 134, 8, 2, 67, 248, 31, 255, 3, 191, 64, 3, 240, 126, 243, 222, 2, 251, 255, 192, 4, 32, 193, 244, 15, 189, 240, 112, 126, 4, 63, 245, 255, 208, 0, 4, 48, 64, 12, 79, 73, 244, 32, 124, 20, 47, 128, 236, 62, 1, 11, 255, 124, 19, 240, 127, 248, 95, 127, 23, 224, 124, 23, 255, 51, 252, 78, 195, 3, 255, 188, 255, 238, 196, 24, 48, 255, 8, 49, 189, 244, 95, 5, 240, 111, 197, 3, 176, 192, 248, 78, 64, 19, 225, 0, 231, 225, 68, 20, 80, 194, 251, 238, 188, 12, 15, 190, 252, 96, 188, 255, 239, 60, 251, 223, 67, 3, 239, 128, 11, 242, 190, 11, 240, 1, 240, 31, 67, 244, 31, 254, 8, 81, 123, 0, 17, 65, 224, 1, 68, 247, 239, 128, 7, 206, 189, 255, 255, 66, 19, 160, 185, 248, 160, 254, 4, 16, 60, 23, 224, 130, 11, 177, 195, 12, 48, 192, 15, 255, 252, 0, 0, 125, 24, 1, 124, 255, 144, 195, 16, 30, 126, 16, 48, 64, 252, 79, 190, 4, 33, 62, 7, 175, 128, 255, 239, 200, 15, 191, 189, 8, 16, 188, 251, 143, 188, 8, 127, 184, 255, 241, 1, 12, 46, 129, 0, 47, 62, 243, 191, 1, 240, 126, 65, 251, 192, 127, 44, 15, 184, 244, 63, 64, 28, 93, 129, 255, 206, 190, 7, 192, 126, 11, 241, 126, 235, 238, 131, 12, 81, 64, 3, 255, 126, 243, 224, 132, 3, 209, 255, 255, 225, 5, 248, 64, 194, 11, 223, 6, 7, 255, 254, 244, 110, 126, 255, 240, 128, 0, 64, 13, 232, 48, 0, 15, 63, 130, 239, 157, 130, 232, 78, 255, 243, 206, 196, 232, 30, 1, 0, 33, 3, 16, 46, 252, 8, 15, 59, 8, 1, 4, 27, 174, 66, 247, 240, 64, 248, 62, 195, 16, 16, 3, 27, 192, 65, 15, 175, 249, 240, 0, 64, 16, 19, 58, 8, 159, 254, 247, 190, 131, 23, 160, 2, 15, 239, 1, 0, 82, 186, 4, 16, 126, 16, 15, 130, 28, 47, 65, 244, 79, 78, 247, 176, 0, 240, 46, 65, 248, 47, 56, 8, 48, 254, 8, 161, 247, 7, 255, 130, 236, 127, 66, 232, 16, 4, 4, 17, 3, 232, 48, 123, 19, 208, 253, 255, 143, 131, 15, 159, 197, 255, 205, 126, 4, 15, 65, 15, 255, 185, 244, 35, 117, 8, 96, 193, 28, 95, 250, 20, 78, 192, 8, 15, 2, 220, 15, 66, 8, 32, 69, 7, 144, 2, 11, 207, 128, 239, 255, 188, 236, 79, 191, 244, 32, 10, 252, 0, 192, 32, 96, 192, 4, 48, 62, 248, 31, 250, 16, 65, 7, 228, 17, 122, 240, 31, 129, 32, 47, 193, 228, 65, 67, 255, 226, 6, 235, 62, 136, 27, 177, 63, 19, 146, 4, 3, 255, 122, 0, 1, 68, 16, 46, 127, 252, 33, 67, 231, 255, 74, 243, 241, 63, 7, 225, 129, 220, 49, 126, 36, 15, 69, 0, 48, 63, 45, 221, 207, 30, 21, 19, 227, 239, 21, 232, 220, 19, 9, 229, 10, 238, 3, 239, 220, 23, 8, 23, 6, 253, 3, 230, 236, 228, 243, 14, 189, 25, 9, 5, 25, 6, 233, 12, 232, 6, 235, 11, 25, 231, 25, 239, 251, 241, 13, 13, 241, 220, 12, 246, 241, 244, 248, 254, 251, 233, 249, 85, 14, 33, 7, 252, 220, 203, 223, 233, 244, 247, 238, 26, 248, 20, 33, 21, 249, 16, 0, 42, 242, 245, 255, 20, 26, 218, 34, 10, 236, 254, 4, 12, 239, 11, 41, 235, 32, 25, 48, 242, 211, 228, 1, 229, 222, 239, 244, 221, 234, 23, 241, 254, 20, 18, 23, 248, 28, 54, 5, 1, 9, 248, 246, 31, 2, 221, 25, 249, 3, 16, 199, 244, 2, 8, 233, 5, 44, 57, 66, 248, 255, 242, 204, 249, 253, 248, 60, 250, 18, 220, 9, 28, 13, 2, 240, 4, 17, 245, 40, 30, 64, 215, 249, 45, 186, 17, 215, 61, 4, 193, 11, 253, 19, 230, 23, 17, 10, 243, 237, 5, 246, 207, 231, 5, 224, 247, 14, 31, 248, 5, 51, 252, 18, 12, 0, 44, 232, 31, 245, 38, 56, 25, 15, 227, 254, 214, 240, 251, 187, 35, 230, 244, 8, 239, 50, 34, 11, 19, 221, 39, 240, 7, 229, 250, 0, 215, 38, 20, 240, 227, 2, 33, 7, 7, 236, 17, 30, 7, 14, 42, 3, 45, 249, 29, 227, 252, 232, 0, 241, 249, 242, 247, 254, 23, 1, 1, 24, 4, 18, 203, 209, 233, 0, 25, 242, 1, 21, 34, 218, 234, 237, 19, 248, 229, 244, 37, 220, 239, 36, 224, 28, 230, 20, 231, 220, 236, 1, 242, 244, 249, 20, 244, 1, 1, 7, 237, 38, 226, 233, 223, 11, 245, 240, 7, 234, 7, 250, 251, 198, 215, 230, 7, 250, 252, 253, 39, 13, 253, 13, 23, 252, 78, 240, 238, 0, 7, 26, 236, 222, 9, 248, 242, 21, 225, 19, 248, 2, 9, 204, 243, 8, 215, 230, 37, 30, 206, 14, 223, 237, 12, 201, 244, 5, 11, 39, 20, 246, 27, 247, 3, 240, 235, 241, 4, 1, 13, 235, 251, 242, 26, 252, 27, 240, 24, 35, 254, 222, 250, 247, 248, 7, 227, 243, 2, 10, 235, 1, 254, 25, 238, 232, 233, 13, 0, 229, 250, 237, 30, 253, 246, 40, 229, 240, 230, 240, 252, 19, 244, 251, 5, 251, 11, 9, 234, 10, 14, 8, 238, 19, 41, 50, 18, 233, 12, 228, 254, 242, 35, 244, 255, 3, 11, 235, 237, 27, 64, 46, 210, 246, 23, 17, 2, 188, 12, 249, 233, 243, 53, 237, 12, 1, 250, 240, 254, 250, 228, 29, 240, 5, 10, 22, 44, 17, 23, 28, 217, 11, 238, 33, 18, 24, 237, 250, 250, 15, 3, 244, 23, 20, 18, 243, 25, 214, 11, 1, 250, 238, 31, 40, 35, 240, 214, 239, 18, 214, 223, 234, 251, 252, 23, 13, 236, 218, 6, 231, 206, 213, 0, 3, 30};
// #endif