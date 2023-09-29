// https://github.com/pnnl/QASMBench/tree/master/large/multiplier_n45
OPENQASM 2.0;
include "qelib1.inc";
qreg q0[45];
creg c0[9];
x q0[40];
x q0[37];
x q0[30];
x q0[29];
x q0[28];
ccx q0[36],q0[27],q0[1];
ccx q0[36],q0[28],q0[4];
ccx q0[36],q0[29],q0[7];
ccx q0[36],q0[30],q0[10];
ccx q0[36],q0[31],q0[13];
ccx q0[36],q0[32],q0[16];
ccx q0[36],q0[33],q0[19];
ccx q0[36],q0[34],q0[22];
ccx q0[36],q0[35],q0[25];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[0],q0[2],q0[3];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[3],q0[5],q0[6];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[6],q0[8],q0[9];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[9],q0[11],q0[12];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[12],q0[14],q0[15];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[15],q0[17],q0[18];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[18],q0[20],q0[21];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[21],q0[23],q0[24];
cx q0[25],q0[26];
cx q0[24],q0[26];
ccx q0[21],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
cx q0[21],q0[23];
ccx q0[18],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
cx q0[18],q0[20];
ccx q0[15],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
cx q0[15],q0[17];
ccx q0[12],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
cx q0[12],q0[14];
ccx q0[9],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
cx q0[9],q0[11];
ccx q0[6],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
cx q0[6],q0[8];
ccx q0[3],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
cx q0[3],q0[5];
ccx q0[0],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
cx q0[0],q0[2];
ccx q0[36],q0[27],q0[1];
ccx q0[36],q0[28],q0[4];
ccx q0[36],q0[29],q0[7];
ccx q0[36],q0[30],q0[10];
ccx q0[36],q0[31],q0[13];
ccx q0[36],q0[32],q0[16];
ccx q0[36],q0[33],q0[19];
ccx q0[36],q0[34],q0[22];
ccx q0[36],q0[35],q0[25];
ccx q0[37],q0[27],q0[4];
ccx q0[37],q0[28],q0[7];
ccx q0[37],q0[29],q0[10];
ccx q0[37],q0[30],q0[13];
ccx q0[37],q0[31],q0[16];
ccx q0[37],q0[32],q0[19];
ccx q0[37],q0[33],q0[22];
ccx q0[37],q0[34],q0[25];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[0],q0[2],q0[3];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[3],q0[5],q0[6];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[6],q0[8],q0[9];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[9],q0[11],q0[12];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[12],q0[14],q0[15];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[15],q0[17],q0[18];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[18],q0[20],q0[21];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[21],q0[23],q0[24];
cx q0[25],q0[26];
cx q0[24],q0[26];
ccx q0[21],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
cx q0[21],q0[23];
ccx q0[18],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
cx q0[18],q0[20];
ccx q0[15],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
cx q0[15],q0[17];
ccx q0[12],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
cx q0[12],q0[14];
ccx q0[9],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
cx q0[9],q0[11];
ccx q0[6],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
cx q0[6],q0[8];
ccx q0[3],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
cx q0[3],q0[5];
ccx q0[0],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
cx q0[0],q0[2];
ccx q0[37],q0[27],q0[4];
ccx q0[37],q0[28],q0[7];
ccx q0[37],q0[29],q0[10];
ccx q0[37],q0[30],q0[13];
ccx q0[37],q0[31],q0[16];
ccx q0[37],q0[32],q0[19];
ccx q0[37],q0[33],q0[22];
ccx q0[37],q0[34],q0[25];
ccx q0[38],q0[27],q0[7];
ccx q0[38],q0[28],q0[10];
ccx q0[38],q0[29],q0[13];
ccx q0[38],q0[30],q0[16];
ccx q0[38],q0[31],q0[19];
ccx q0[38],q0[32],q0[22];
ccx q0[38],q0[33],q0[25];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[0],q0[2],q0[3];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[3],q0[5],q0[6];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[6],q0[8],q0[9];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[9],q0[11],q0[12];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[12],q0[14],q0[15];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[15],q0[17],q0[18];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[18],q0[20],q0[21];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[21],q0[23],q0[24];
cx q0[25],q0[26];
cx q0[24],q0[26];
ccx q0[21],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
cx q0[21],q0[23];
ccx q0[18],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
cx q0[18],q0[20];
ccx q0[15],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
cx q0[15],q0[17];
ccx q0[12],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
cx q0[12],q0[14];
ccx q0[9],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
cx q0[9],q0[11];
ccx q0[6],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
cx q0[6],q0[8];
ccx q0[3],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
cx q0[3],q0[5];
ccx q0[0],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
cx q0[0],q0[2];
ccx q0[38],q0[27],q0[7];
ccx q0[38],q0[28],q0[10];
ccx q0[38],q0[29],q0[13];
ccx q0[38],q0[30],q0[16];
ccx q0[38],q0[31],q0[19];
ccx q0[38],q0[32],q0[22];
ccx q0[38],q0[33],q0[25];
ccx q0[39],q0[27],q0[10];
ccx q0[39],q0[28],q0[13];
ccx q0[39],q0[29],q0[16];
ccx q0[39],q0[30],q0[19];
ccx q0[39],q0[31],q0[22];
ccx q0[39],q0[32],q0[25];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[0],q0[2],q0[3];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[3],q0[5],q0[6];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[6],q0[8],q0[9];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[9],q0[11],q0[12];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[12],q0[14],q0[15];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[15],q0[17],q0[18];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[18],q0[20],q0[21];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[21],q0[23],q0[24];
cx q0[25],q0[26];
cx q0[24],q0[26];
ccx q0[21],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
cx q0[21],q0[23];
ccx q0[18],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
cx q0[18],q0[20];
ccx q0[15],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
cx q0[15],q0[17];
ccx q0[12],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
cx q0[12],q0[14];
ccx q0[9],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
cx q0[9],q0[11];
ccx q0[6],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
cx q0[6],q0[8];
ccx q0[3],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
cx q0[3],q0[5];
ccx q0[0],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
cx q0[0],q0[2];
ccx q0[39],q0[27],q0[10];
ccx q0[39],q0[28],q0[13];
ccx q0[39],q0[29],q0[16];
ccx q0[39],q0[30],q0[19];
ccx q0[39],q0[31],q0[22];
ccx q0[39],q0[32],q0[25];
ccx q0[40],q0[27],q0[13];
ccx q0[40],q0[28],q0[16];
ccx q0[40],q0[29],q0[19];
ccx q0[40],q0[30],q0[22];
ccx q0[40],q0[31],q0[25];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[0],q0[2],q0[3];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[3],q0[5],q0[6];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[6],q0[8],q0[9];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[9],q0[11],q0[12];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[12],q0[14],q0[15];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[15],q0[17],q0[18];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[18],q0[20],q0[21];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[21],q0[23],q0[24];
cx q0[25],q0[26];
cx q0[24],q0[26];
ccx q0[21],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
cx q0[21],q0[23];
ccx q0[18],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
cx q0[18],q0[20];
ccx q0[15],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
cx q0[15],q0[17];
ccx q0[12],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
cx q0[12],q0[14];
ccx q0[9],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
cx q0[9],q0[11];
ccx q0[6],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
cx q0[6],q0[8];
ccx q0[3],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
cx q0[3],q0[5];
ccx q0[0],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
cx q0[0],q0[2];
ccx q0[40],q0[27],q0[13];
ccx q0[40],q0[28],q0[16];
ccx q0[40],q0[29],q0[19];
ccx q0[40],q0[30],q0[22];
ccx q0[40],q0[31],q0[25];
ccx q0[41],q0[27],q0[16];
ccx q0[41],q0[28],q0[19];
ccx q0[41],q0[29],q0[22];
ccx q0[41],q0[30],q0[25];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[0],q0[2],q0[3];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[3],q0[5],q0[6];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[6],q0[8],q0[9];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[9],q0[11],q0[12];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[12],q0[14],q0[15];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[15],q0[17],q0[18];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[18],q0[20],q0[21];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[21],q0[23],q0[24];
cx q0[25],q0[26];
cx q0[24],q0[26];
ccx q0[21],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
cx q0[21],q0[23];
ccx q0[18],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
cx q0[18],q0[20];
ccx q0[15],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
cx q0[15],q0[17];
ccx q0[12],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
cx q0[12],q0[14];
ccx q0[9],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
cx q0[9],q0[11];
ccx q0[6],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
cx q0[6],q0[8];
ccx q0[3],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
cx q0[3],q0[5];
ccx q0[0],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
cx q0[0],q0[2];
ccx q0[41],q0[27],q0[16];
ccx q0[41],q0[28],q0[19];
ccx q0[41],q0[29],q0[22];
ccx q0[41],q0[30],q0[25];
ccx q0[42],q0[27],q0[19];
ccx q0[42],q0[28],q0[22];
ccx q0[42],q0[29],q0[25];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[0],q0[2],q0[3];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[3],q0[5],q0[6];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[6],q0[8],q0[9];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[9],q0[11],q0[12];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[12],q0[14],q0[15];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[15],q0[17],q0[18];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[18],q0[20],q0[21];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[21],q0[23],q0[24];
cx q0[25],q0[26];
cx q0[24],q0[26];
ccx q0[21],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
cx q0[21],q0[23];
ccx q0[18],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
cx q0[18],q0[20];
ccx q0[15],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
cx q0[15],q0[17];
ccx q0[12],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
cx q0[12],q0[14];
ccx q0[9],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
cx q0[9],q0[11];
ccx q0[6],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
cx q0[6],q0[8];
ccx q0[3],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
cx q0[3],q0[5];
ccx q0[0],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
cx q0[0],q0[2];
ccx q0[42],q0[27],q0[19];
ccx q0[42],q0[28],q0[22];
ccx q0[42],q0[29],q0[25];
ccx q0[43],q0[27],q0[22];
ccx q0[43],q0[28],q0[25];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[0],q0[2],q0[3];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[3],q0[5],q0[6];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[6],q0[8],q0[9];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[9],q0[11],q0[12];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[12],q0[14],q0[15];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[15],q0[17],q0[18];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[18],q0[20],q0[21];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[21],q0[23],q0[24];
cx q0[25],q0[26];
cx q0[24],q0[26];
ccx q0[21],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
cx q0[21],q0[23];
ccx q0[18],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
cx q0[18],q0[20];
ccx q0[15],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
cx q0[15],q0[17];
ccx q0[12],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
cx q0[12],q0[14];
ccx q0[9],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
cx q0[9],q0[11];
ccx q0[6],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
cx q0[6],q0[8];
ccx q0[3],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
cx q0[3],q0[5];
ccx q0[0],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
cx q0[0],q0[2];
ccx q0[43],q0[27],q0[22];
ccx q0[43],q0[28],q0[25];
ccx q0[44],q0[27],q0[25];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[0],q0[2],q0[3];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[3],q0[5],q0[6];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[6],q0[8],q0[9];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[9],q0[11],q0[12];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[12],q0[14],q0[15];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[15],q0[17],q0[18];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[18],q0[20],q0[21];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[21],q0[23],q0[24];
cx q0[25],q0[26];
cx q0[24],q0[26];
ccx q0[21],q0[23],q0[24];
cx q0[22],q0[23];
ccx q0[22],q0[23],q0[24];
cx q0[22],q0[23];
cx q0[21],q0[23];
ccx q0[18],q0[20],q0[21];
cx q0[19],q0[20];
ccx q0[19],q0[20],q0[21];
cx q0[19],q0[20];
cx q0[18],q0[20];
ccx q0[15],q0[17],q0[18];
cx q0[16],q0[17];
ccx q0[16],q0[17],q0[18];
cx q0[16],q0[17];
cx q0[15],q0[17];
ccx q0[12],q0[14],q0[15];
cx q0[13],q0[14];
ccx q0[13],q0[14],q0[15];
cx q0[13],q0[14];
cx q0[12],q0[14];
ccx q0[9],q0[11],q0[12];
cx q0[10],q0[11];
ccx q0[10],q0[11],q0[12];
cx q0[10],q0[11];
cx q0[9],q0[11];
ccx q0[6],q0[8],q0[9];
cx q0[7],q0[8];
ccx q0[7],q0[8],q0[9];
cx q0[7],q0[8];
cx q0[6],q0[8];
ccx q0[3],q0[5],q0[6];
cx q0[4],q0[5];
ccx q0[4],q0[5],q0[6];
cx q0[4],q0[5];
cx q0[3],q0[5];
ccx q0[0],q0[2],q0[3];
cx q0[1],q0[2];
ccx q0[1],q0[2],q0[3];
cx q0[1],q0[2];
cx q0[0],q0[2];
ccx q0[44],q0[27],q0[25];
//measure q0[2] -> c0[0];
//measure q0[5] -> c0[1];
//measure q0[8] -> c0[2];
//measure q0[11] -> c0[3];
//measure q0[14] -> c0[4];
//measure q0[17] -> c0[5];
//measure q0[20] -> c0[6];
//measure q0[23] -> c0[7];
//measure q0[26] -> c0[8];