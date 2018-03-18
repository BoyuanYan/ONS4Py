|cluster|wave_num|network|network_factor|rou|miu|base_lr|max_iter|end|result|best now|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|local|5|simplestnet||8|120|7e-4|300|True||0.10|
|20|5|simplenet||8|120|7e-4|300|True|0.10|0.10
|20|10|simplenet||8|300|7e-4|3000|False|||
|20|10|expandsimplenet|expand=2|8|300|2e-4|3000|False||0.21|
|20|10|expandsimplenet|expand=4|8|300|7e-4|3000|False||0,33|
|20|10|expandsimplenet|expand=2|8|300|7e-5|3000|False||0.35|
|30|15|expandsimplenet|expand=4|8|480|1e-3|3000|False||0.4|
|30|10|expandsimplenet|expand=4|8|300|2e-4|3000|False||0.25|



|cluster|wave_num|expand_factor|base_lr|work?
|:----|:----|:----|:----|:----|
|20|10|2|2e-4|Yes|
|20|10|2|7e-4|
|20|10|2|7e-5|Yes|
|20|10|3|2e-4|
|20|10|3|7e-4|
|20|10|4|7e-4|Yes|
|20|15|3|7e-4|
|20|15|3|2e-4|
|20|15|3|1e-4|
|20|20|3|7e-4|
|20|25|3|7e-4|
|20|30|3|7e-4|
|20|40|3|7e-4|
|30|10|4|1e-3|
|30|10|4|2e-4|Yes|
|30|15|4|1e-3|
|30|15|4|2e-4|
|30|20|4|2e-4|
