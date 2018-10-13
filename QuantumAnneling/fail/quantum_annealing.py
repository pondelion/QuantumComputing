import numpy as np
import matplotlib.pyplot as plt


class System():
    def __init__(self, temperature, kb=1):
        self._temperature = temperature
        self._kb = kb

    # 温度の設定
    def set_temperature(self, temperature):
        self._temperature = temperature

    # 温度の取得
    def get_temperature(self):
        return self._temperature


class SpinSystem(System):
    def __init__(self, spin_num_col, spin_num_row, interactions, kb=1, temperature=30):
        super().__init__(temperature, kb)
        self._spin_num_col = spin_num_col
        self._spin_num_row = spin_num_row
        self._interactions = interactions
        self._spin_map = self._create_spinmap(spin_num_row, spin_num_col)

    # 温度の設定
    def set_temperature(self, temperature):
        self._temperature = temperature

    # 温度の取得
    def get_temperature(self):
        return self._temperature

    # [i, j]のスピンを更新
    def _update_spin(self, i, j):
        # スピンが↑を向く確率
        up_prob = np.exp(-self._calc_spin_energy(i, j, 0)/(self._kb*self._temperature)) / \
                  (np.exp(-self._calc_spin_energy(i, j, 0)/(self._kb*self._temperature)) + \
                  np.exp(-self._calc_spin_energy(i, j, np.pi)/(self._kb*self._temperature)))
        rand = np.random.rand()
        if up_prob > rand:
            self._spin_map[i][j] = 0  # ↑向き
        else:
            self._spin_map[i][j] = 1  # ↓向き

    # スピンをランダムにnum_spin個更新
    def update_spins(self, num_spin):
        for i, j in zip(np.random.randint(0, self._spin_num_row, num_spin), np.random.randint(0, self._spin_num_col, num_spin)):
            self._update_spin(i, j)

    # [i, j]のスピンがphi方向向いているときのエネルギーを計算
    def _calc_spin_energy(self, i, j, phi):
        energy = 0
        try:
            energy += -self._interactions[i*j][(i-1)*j] * np.cos(phi - np.pi*self._spin_map[i-1][j])
        except IndexError as e:
            pass
        try:
            energy += -self._interactions[i*j][(i+1)*j] * np.cos(phi - np.pi*self._spin_map[i+1][j])
        except IndexError as e:
            pass
        try:
            energy += -self._interactions[i*j][i*(j+1)] * np.cos(phi - np.pi*self._spin_map[i][j+1])
        except IndexError as e:
            pass
        try:
            energy += -self._interactions[i*j][i*(j-1)] * np.cos(phi - np.pi*self._spin_map[i][j-1])
        except IndexError as e:
            pass
        return 0.5 * energy

    # 全エネルギーを計算する
    def calc_system_energy(self):
        energy = 0
        for i in range(self._spin_num_row):
            for j in range(self._spin_num_col):
                    energy += self._calc_spin_energy(i, j, np.pi if self._spin_map[i][j] == 1 else 0)
        return energy

    # spin_num_row × spin_num_colのスピンマップを作成
    def _create_spinmap(self, spin_num_row, spin_num_col):
        spin_map = np.zeros([spin_num_row, spin_num_col])
        for i in range(spin_num_row):
            for j in range(spin_num_col):
                spin_map[i][j] = 0 if np.random.rand() <= 0.5 else 1
        return spin_map

    # スピンマップを初期化する
    def initialize_spins(self):
        self._spin_map = self._create_spinmap(spin_num_row, spin_num_col)

    def get_spinmap(self):
        return self._spin_map

class QuantumSpinSystem(System):
    def __init__(self, spin_num_col, spin_num_row, spin_systems, temperature, interactions, kb=1, transverse_field=10):
        super().__init__(temperature, kb)
        self._spin_num_col = spin_num_col
        self._spin_num_row = spin_num_row
        self._spin_systems = spin_systems
        self._transverse_field = transverse_field
        self._interactions = interactions

    def set_temperature(self, temperature):
        self._temperature = temperature
        for spin_system in self._spin_systems:
            spin_system.set_temperature(temperature)

    # 横磁場を設定
    def set_transverse_field(self, transverse_field):
        self._transverse_field = transverse_field

    # 横磁場による相互作用の係数を計算
    def _transverse_field_interaction(self):
        return 0.5 * self._kb * self._temperature * np.log(1/np.tanh(self._transverse_field / (self._kb * self._temperature * len(self._spin_systems))))

    # [i, j, k]のスピンを更新
    def _update_spin(self, i, j, k):
        # スピンが↑を向く確率
        up_prob = np.exp(-self._calc_spin_energy(i, j, k, 0)/(self._kb*self._temperature)) / \
                  (np.exp(-self._calc_spin_energy(i, j, k, 0)/(self._kb*self._temperature)) + \
                  np.exp(-self._calc_spin_energy(i, j, k, np.pi)/(self._kb*self._temperature)))
        rand = np.random.rand()
        if up_prob > rand:
            self._spin_systems[k].get_spinmap()[i][j] = 0  # ↑向き
        else:
            self._spin_systems[k].get_spinmap()[i][j] = 1  # ↓向き

    # スピンをランダムにnum_spin個更新
    def update_spins(self, num_spin):
        for i, j, k in zip(np.random.randint(0, self._spin_num_row, num_spin),
                           np.random.randint(0, self._spin_num_col, num_spin),
                           np.random.randint(0, len(self._spin_systems), num_spin)):
            self._update_spin(i, j, k)

    # [i, j, k]のスピンがphi方向向いているときのエネルギーを計算
    def _calc_spin_energy(self, i, j, k, phi):
        energy = 0
        # ij(スピンマップ)面方向の交換相互作用によるエネルギー計算
        try:
            energy += -self._interactions[i*j][(i-1)*j] * np.cos(phi - np.pi*self._spin_systems[k].get_spinmap()[i-1][j])
        except IndexError as e:
            pass
        try:
            energy += -self._interactions[i*j][(i+1)*j] * np.cos(phi - np.pi*self._spin_systems[k].get_spinmap()[i+1][j])
        except IndexError as e:
            pass
        try:
            energy += -self._interactions[i*j][i*(j+1)] * np.cos(phi - np.pi*self._spin_systems[k].get_spinmap()[i][j+1])
        except IndexError as e:
            pass
        try:
            energy += -self._interactions[i*j][i*(j-1)] * np.cos(phi - np.pi*self._spin_systems[k].get_spinmap()[i][j-1])
        except IndexError as e:
            pass
        energy *= 0.5
        # k方向の相互作用(横磁場)によるエネルギー計算
        s1 = 1 if self._spin_systems[k].get_spinmap()[i][j] == 0 else -1
        #min_k = np.argmin(self.get_system_energies())
        try:
            s2 = 1 if self._spin_systems[k-1].get_spinmap()[i][j] == 0 else -1
            energy += -self._transverse_field_interaction() * s1 * s2
        except IndexError as e:
            pass
        try:
            s2 = 1 if self._spin_systems[k+1].get_spinmap()[i][j] == 0 else -1
            energy += -self._transverse_field_interaction() * s1 * s2
        except IndexError as e:
            pass
        return energy

    # 各スピンマップ(トロッタ？)のエネルギーを配列で返す
    def get_system_energies(self):
        return [spin_system.calc_system_energy() for spin_system in self._spin_systems]

# 100x100個のスピンマップ
SPIN_COL_NUM = 100
SPIN_ROW_NUM = 100

INTERACTIONS = np.random.randn(
    SPIN_COL_NUM * SPIN_ROW_NUM, SPIN_COL_NUM * SPIN_ROW_NUM
)

if __name__ == '__main__':
    temperature = 1000  # 初期温度
    spin_system = SpinSystem(
        spin_num_col=SPIN_COL_NUM,
        spin_num_row=SPIN_ROW_NUM,
        interactions=INTERACTIONS,
        temperature=temperature
    )
    trotta_num = 10  # トロッタ数
    quantum_spin_system = QuantumSpinSystem(
        spin_num_col=SPIN_COL_NUM,
        spin_num_row=SPIN_ROW_NUM,
        temperature=temperature,
        interactions=INTERACTIONS,
        spin_systems=[SpinSystem(
                spin_num_col=SPIN_COL_NUM,
                spin_num_row=SPIN_ROW_NUM,
                interactions=INTERACTIONS,
                temperature=temperature
            ) for i in range(trotta_num)]
    )
    temperature_hist = []
    step_hist = []
    energy_hist = []
    energy2_hist = [[] for i in range(trotta_num)]
    update_num = 10
    step = 0
    delta_t = 1
    # 10Kまでdelta_t刻みで温度下げる
    while temperature > 10:
        # シミュレーテッドアニーリング系の温度を設定
        spin_system.set_temperature(temperature)
        # 量子アニーリング系の温度を設定
        quantum_spin_system.set_temperature(temperature)
        # 横磁場を設定。(温度の1000分の1とした)
        quantum_spin_system.set_transverse_field(temperature/1000)
        temperature -= delta_t
        # シミュレーテッドアニーリング系のスピンをupdate_num個更新
        spin_system.update_spins(num_spin=update_num)
        # 量子アニーリング系のスピンをupdate_num個更新
        quantum_spin_system.update_spins(num_spin=update_num)
        step += update_num
        energy = spin_system.calc_system_energy()
        temperature_hist.append(temperature)
        step_hist.append(step)
        energy_hist.append(energy)
        energy2 = quantum_spin_system.get_system_energies()
        for i, ener in enumerate(energy2):
            energy2_hist[i].append(ener)
        print('{}K : {}'.format(temperature, energy))
        print('quantum energy : ', energy2)
        print('average quantum energy : ', sum(energy2)/len(energy2))
        print(' ')

    plt.plot(step_hist, energy_hist, linestyle='dashed')
    for energy in energy2_hist:
        plt.plot(step_hist, energy)
    plt.show()
