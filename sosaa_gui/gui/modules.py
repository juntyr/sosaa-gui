def init_modules_gui(self):
    def changeFlagEmis(flag_emis):
        self.dt_emis.setEnabled(flag_emis)
        self.label_dt_emis.setEnabled(flag_emis)
        self.group_output_list_emi.setEnabled(flag_emis)

    self.flag_emis.stateChanged.connect(changeFlagEmis)

    def changeFlagChem(flag_chem):
        self.dt_chem.setEnabled(flag_chem)
        self.label_dt_chem.setEnabled(flag_chem)
        self.label_chem_dir.setEnabled(flag_chem)
        self.chem_dir.setEnabled(flag_chem)
        self.browse_chem.setEnabled(flag_chem)

    self.flag_chem.clicked.connect(changeFlagChem)

    def changeFlagGasDryDep(flag_gasdrydep):
        self.dt_depo.setEnabled(flag_gasdrydep)
        self.label_dt_depo.setEnabled(flag_gasdrydep)
        self.group_output_list_Vd.setEnabled(flag_gasdrydep)

    self.flag_gasdrydep.stateChanged.connect(changeFlagGasDryDep)

    def changeFlagAero(flag_aero):
        self.dt_aero.setEnabled(flag_aero)
        self.label_dt_aero.setEnabled(flag_aero)
        self.dt_uhma.setEnabled(flag_aero)
        self.label_dt_uhma.setEnabled(flag_aero)
        self.aero_start_label.setEnabled(flag_aero)
        self.aero_start_offset.setEnabled(flag_aero)
        self.aero_start_date.setEnabled(flag_aero)

    self.flag_aero.clicked.connect(changeFlagAero)
