# Petunjuk Penggunaan Program
- Pada folder yang mengandung `Makefile`, Ketik `make` pada terminal.
- Program akan dikompilasi.
- Setelah itu, jalankan program dengan perintah `./prog <N>` pada direktori yang terdapat Makefile.
- Contoh: `./prog 100` akan menjalankan program dengan N (jumlah node) = 100.
- Akan keluar file output dengan nama `output-<N>.txt` pada direktori yang terdapat Makefile.

# Pembagian Tugas
- Fata Nugraha (13517109) mengerjakan fungsi dijkstra yang berjalan pada gpu dan main program.
- Edward Alexander jaya (13517115) mengerjakan fungsi dijkstra yang berjalan pada gpu dan fungsi file eksternal.

# Laporan Pengerjaan
#### Deskripsi Solusi Paralel
- Terdapat gridDim dan blockDim pada program kami. Perhatikan bahwa 1 grid mempunyai block sebanyak gridDim dan 1 block mempunyai thread sebanyak blockDim.
- gridDim yang dipakai adalah sejumlah N (jumlah node) dan blockDim yang dipakai adalah sejumlah N (jumlah node) untuk inisialisasi array visited. Perhatikan bahwa 1 thread akan mengurus array visited[sourceVertex][destVertex].
- gridDim yang dipakai adalah sejumlah N (jumlah node) dan blockDim yang dipakai adalah sejumlah 1 untuk menjalankan algoritma dijkstra. Artinya, terdapat N block yang menjalankan dijkstra. Setiap block mengeksekusi dijkstra dari sebuah node ke semua node lain.

#### Analisis Solusi Paralel

Alokasikan semua array 1D yang dibutuhkan pada CPU dan GPU. Pada GPU, alokasi dilakukan dengan menggunakan:
    
    cudaMalloc((void **) &gpuGraph, (sizeof(int) * N * N));
    cudaMalloc((void **) &gpuVisited, (sizeof(bool) * N * N));
    cudaMalloc((void **) &gpuResult, (sizeof(int) * N * N));

Copy graph dari CPU menuju GPU menggunakan:

    cudaMemcpy(gpuGraph, cpuGraph, (sizeof(int) * N * N), cudaMemcpyHostToDevice);

Kemudian lakukan inisialisasi array 1D gpuVisited untuk menentukan node yang sudah dikunjungi.
Perhakan dalam inisialisasi array 1D tersebut, dibutuhkan gridDim = N dan blockDim = N.
Satu thread adalah representasi satu index array 1D gpuVisited.

    __global__ void initializeVisited(int *result, bool *visited) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        visited[index] = false;
      
        if ( index == ((blockDim.x * blockIdx.x) + blockIdx.x)) {
          result[index] = 0;
        } else {
          else result[index] = INT_MAX;
        }
        
    }


Setelah itu, lakukan dijkstra dengan gridDim = N dan blockDim = 1.
Karena blockDim = 1, maka representasi index adalah

      N * blockIdx.x;

Perhatikan bahwa hasil dari dijkstra disimpan ke dalam array 1D gpuResult pada fungsi

      __global__ void dijkstra(int *graph, int *result, bool* visited, int N)

Copy gpuResult dari GPU kepada cpuResult pada CPU:

      cudaMemcpy(result, gpuResult, (sizeof(int) * N * N), cudaMemcpyDeviceToHost);
      

#### Jumlah Thread yang Digunakan
- Ada N buah thread yang digunakan, yang tersebar dalam N buah block dengan 1 buah thread di masing-masing blocknya. Alasannya, setiap thread tidak perlu saling mengetahui keadaan di thread lain, sehingga thread dapat dipisahkan pada block-block yang berbeda.

#### Pengukuran Kinerja untuk tiap Kasus Uji
Berikut adalah hasil pengujian yang dikerjakan pada server 167.205.32.100:
- **N = 100**

  | Tipe | Percobaan 1 | Percobaan 2 | Percobaan 3 |
  |---|--- |---|---|
  | Serial   | 19318.000000 µs  | 23879.000000 µs   | 21159.000000 µs|
  | Paralel | 17057.439453 µs | 17050.335938 µs | 17051.679688 µs|

- **N = 500**

  | Tipe  |  Percobaan 1 | Percobaan 2  | Percobaan 3  |
  |---|---|---|---|
  | Serial |  1525218.000000 µs |  1505172.000000 µs |  1461822.000000 µs |
  | Paralel  |  427482.687500 µs |  446398.000000 µs |  427946.062500 µs |
- **N = 1000**

  | Tipe  |  Percobaan 1 | Percobaan 2  | Percobaan 3  |
  |---|---|---|---|
  | Serial | 10383358.000000 µs | 10314490.000000 µs | 10425789.000000 µs |
  | Paralel  | 4646215.500000 µs |  4647795.500000 µs | 4646911.500000 µs |
- **N = 3000**

  | Tipe  |  Percobaan 1 | Percobaan 2  | Percobaan 3  |
  |---|---|---|---|
  | Serial | 300571112.000000 µs  |  300935149.000000 µs |  306387030.000000 µs|
  | Paralel  | 110913656.000000 µs | 103516760.000000 µs |  136463296.000000 µs|

#### Analisis Perbandingan Kinerja Serial dan Paralel
- Pada program serial, hanya ada satu proses yang menjalankan program dijkstra. Pada program paralel dengan GPU, program dapat dijalankan pada N proses yang berbeda, tetapi ada overhead berupa waktu transfer data dari CPU ke GPU.
- Untuk program dengan N kecil, waktu eksekusi program secara serial dan waktu eksekusi program secara paralel hampir sama. Proses di GPU tentu lebih cepat dari CPU, karena ada N buah pekerjaan yang dilakukan bersamaan, tetapi overhead transfer data dari CPU ke GPU membuat waktu eksekusi totalnya hampir sama
- Untuk program dengan N besar, waktu eksekusinya di GPU jauh lebih cepat daripada waktu eksekusi di CPU. Alasannya, waktu transfer data kira-kira berderajat O(n) sedangkan waktu eksekusi dijkstra untuk semua node berderajat O(n^3), sehingga overhead waktu transfer data ke GPU lebih kecil daripada selisih waktu eksekusi di CPU dan GPU.