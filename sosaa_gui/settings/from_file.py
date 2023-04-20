from pathlib import Path

from PyQt5 import QtCore, QtWidgets


def update_gui_from_settings(settings, gui, raw):
    _update_gui_from_main_settings(settings, gui)
    _update_gui_from_flag_settings(settings, gui)
    _update_gui_from_time_settings(settings, gui)
    _update_gui_from_output_settings(settings, gui)
    _update_gui_from_custom_settings(settings, gui)
    _update_gui_from_compile_settings(settings, gui)
    _update_gui_from_run_settings(settings, gui)
    _update_gui_from_rsm_settings(settings, gui)
    _update_gui_from_raw_settings(gui, raw)


def _update_gui_from_main_settings(settings, gui):
    main = settings.get("nml_main", dict())
    guis = settings.get("nml_gui", dict())

    # Pretty-print the main dir if relative to the working dir
    main_dir = str(Path(main.get("work_dir", "./sosaa")).resolve())
    if main_dir.startswith(str(Path.cwd())):
        main_dir = f"./{Path(main_dir).relative_to(Path.cwd())}"
    gui.main_dir.setText(main_dir)

    main_dir = Path(main.get("work_dir", "./sosaa")).resolve()

    # Pretty-print the code dir if relative to the main dir
    code_dir = main.get("code_dir", "src")
    if code_dir.startswith(str(main_dir)):
        code_dir = f"{Path(code_dir).relative_to(main_dir)}"
    gui.code_dir.setText(code_dir)

    # Pretty-print the case dir if relative to the main dir
    case_dir = main.get("case_dir", "casedir")
    if case_dir.startswith(str(main_dir)):
        case_dir = f"{Path(case_dir).relative_to(main_dir)}"
    gui.case_dir.setText(case_dir)

    gui.casename_dir.setText(guis.get("main_casename_dir", "sample_aerosol"))

    # Split the chem_dir into the outer dir and the chemistry name
    chem_dir = main.get("chem_dir", None)
    chemall_dir = guis.get("main_chemall_dir", None)
    chemname_dir = guis.get("main_chemname_dir", None)

    if chemall_dir is None or chemname_dir is None:
        if chem_dir is not None:
            chemall_dir = str(Path(chem_dir).parent)
            chemname_dir = Path(chem_dir).name
        else:
            chemall_dir = "chemistry"
            chemname_dir = "sample_aerosol"

    # Pretty-print the chem dir if relative to the main dir
    if chemall_dir.startswith(str(main_dir)):
        chemall_dir = f"{Path(chemall_dir).relative_to(main_dir)}"

    gui.chem_dir.setText(chemall_dir)
    gui.chemname_dir.setText(chemname_dir)

    # Pretty-print the input dir if relative to the working dir
    input_dir = str(Path(main.get("input_dir", "")).resolve())
    if input_dir.startswith(str(Path.cwd())):
        input_dir = f"./{Path(input_dir).relative_to(Path.cwd())}"
    gui.input_dir.setText(input_dir)

    # Pretty-print the output dir if relative to the working dir
    output_dir = str(Path(main.get("output_dir", "")).resolve())
    if output_dir.startswith(str(Path.cwd())):
        output_dir = f"./{Path(output_dir).relative_to(Path.cwd())}"
    gui.output_dir.setText(output_dir)
    gui.currentOutputPath.setText(output_dir)


def _update_gui_from_flag_settings(settings, gui):
    flag = settings.get("nml_flag", dict())
    aer = settings.get("aer_flag", dict()).get("options", dict())

    # flag_emis affects flag_emis, dt_emis, and output_list_emi
    flag_emis = flag.get("flag_emis", 2) != 0
    gui.flag_emis.setChecked(flag_emis)
    gui.dt_emis.setEnabled(flag_emis)
    gui.label_dt_emis.setEnabled(flag_emis)
    gui.group_output_list_emi.setEnabled(flag_emis)

    # Note: Soil emissions are currently always disabled
    gui.flag_emis_soil.setChecked(False)

    # flag_chem affects flag_chem, dt_chem, and chem_dir
    flag_chem = flag.get("flag_chem", 1) != 0
    gui.flag_chem.setChecked(flag_chem)
    gui.dt_chem.setEnabled(flag_chem)
    gui.label_dt_chem.setEnabled(flag_chem)
    gui.label_chem_dir.setEnabled(flag_chem)
    gui.chem_dir.setEnabled(flag_chem)
    gui.browse_chem.setEnabled(flag_chem)

    flag_mix_chem = flag.get("flag_mix_chem", 1) != 0
    gui.flag_mix_chem.setChecked(flag_mix_chem)

    # flag_gasdrydep affects flag_gasdrydep, dt_depo, and output_list_Vd
    flag_gasdrydep = flag.get("flag_gasdrydep", 0) != 0
    gui.flag_gasdrydep.setChecked(flag_gasdrydep)
    gui.dt_depo.setEnabled(flag_gasdrydep)
    gui.label_dt_depo.setEnabled(flag_gasdrydep)
    gui.group_output_list_Vd.setEnabled(flag_gasdrydep)

    # flag_aero affects flag_aero, dt_aero, dt_uhma, and aero_start_date
    flag_aero = flag.get("flag_aero", 1) != 0
    gui.flag_aero.setChecked(flag_aero)
    gui.dt_aero.setEnabled(flag_aero)
    gui.label_dt_aero.setEnabled(flag_aero)
    gui.dt_uhma.setEnabled(flag_aero)
    gui.label_dt_uhma.setEnabled(flag_aero)
    gui.aero_start_label.setEnabled(flag_aero)
    gui.aero_start_offset.setEnabled(flag_aero)
    gui.aero_start_date.setEnabled(flag_aero)

    flag_mix_aero = flag.get("flag_mix_aero", 1) != 0
    gui.flag_mix_aero.setChecked(flag_mix_aero)

    aer_nucleation = aer.get("nucleation", True)
    gui.aer_nucleation.setChecked(aer_nucleation)

    aer_condensation = aer.get("condensation", True)
    gui.aer_condensation.setChecked(aer_condensation)

    aer_coagulation = aer.get("coagulation", True)
    gui.aer_coagulation.setChecked(aer_coagulation)

    aer_dry_deposition = aer.get("dry_deposition", False)
    gui.aer_dry_deposition.setChecked(aer_dry_deposition)

    # Note: Aerosol wet deposition is not yet implemented
    gui.aer_wet_deposition.setChecked(False)

    aer_snow_scavenge = aer.get("snow_scavenge", False)
    gui.aer_snow_scavenge.setChecked(aer_snow_scavenge)

    flag_vapor = flag.get("flag_vapor", 0) != 0
    gui.flag_vapor.setChecked(flag_vapor)

    flag_debug = flag.get("flag_debug", 0) != 0
    gui.flag_debug.setChecked(flag_debug)


def _update_gui_from_time_settings(settings, gui):
    time = settings.get("nml_time", dict())

    gui.start_date.setDateTime(
        QtCore.QDateTime.fromString(
            ",".join(
                f"{s:02}"
                for s in time.get("start_date", [2023, 4, 1, 0, 0, 0])
            ),
            "yyyy,MM,dd,HH,mm,ss",
        )
    )
    gui.end_date.setDateTime(
        QtCore.QDateTime.fromString(
            ",".join(
                f"{s:02}" for s in time.get("end_date", [2023, 4, 1, 0, 0, 0])
            ),
            "yyyy,MM,dd,HH,mm,ss",
        )
    )

    # Note: A negative value would be misinterpreted afterwards
    gui.trajectory_duration.setValue(
        abs(gui.start_date.dateTime().daysTo(gui.end_date.dateTime()))
    )
    gui.trajectory_duration.lastValue = gui.trajectory_duration.value()

    fullDays = gui.trajectory_duration.value()

    # Positive duration: end_date = ceil(start_date + full_days)
    gui.start_date.setEnabled(True)
    gui.end_date.setEnabled(False)

    gui.end_date.setDate(
        gui.start_date.date().addDays(
            fullDays + (gui.start_date.time() > QtCore.QTime(0, 0))
        )
    )

    aero_start_date = QtCore.QDateTime.fromString(
        ",".join(
            f"{s:02}"
            for s in time.get("aero_start_date", [2000, 1, 1, 0, 0, 0])
        ),
        "yyyy,MM,dd,HH,mm,ss",
    )
    secs_from_start = gui.start_date.dateTime().secsTo(aero_start_date)

    if secs_from_start < 0:
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText("Eager early aerosols")
        msg.setInformativeText(
            "The aerosol simulation cannot start before the simulation itself."
            " Its starting time will be clamped to the simulation start."
        )
        msg.setWindowTitle("Warning loading INITFILE")
        msg.exec_()

    # Calculate aero_start_date based on aero_start_offset
    #  based on aero_start_date
    gui.aero_start_offset.setValue(secs_from_start // (60 * 60))
    gui.aero_start_date.setText(
        gui.start_date.dateTime()
        .addSecs(gui.aero_start_offset.value() * 60 * 60)
        .toString(" dd/MM/yyyy HH:mm:ss")
    )

    gui.dt_obs.setValue(float(time.get("dt_obs", 1800)))
    gui.dt_mete.setValue(float(time.get("dt_mete", 10)))
    gui.dt_emis.setValue(float(time.get("dt_emis", 60)))
    gui.dt_chem.setValue(float(time.get("dt_chem", 60)))
    gui.dt_depo.setValue(float(time.get("dt_depo", 60)))
    gui.dt_aero.setValue(float(time.get("dt_aero", 60)))
    gui.dt_uhma.setValue(float(time.get("dt_uhma", 10)))


def _update_gui_from_output_settings(settings, gui):
    output = settings.get("nml_output", dict())
    guis = settings.get("nml_gui", dict())

    gui.output_list_spc.setPlainText(
        ", ".join(
            s.strip()
            for s in output.get("output_list_spc", "").split(",")
            if len(s.strip()) > 0
        )
    )
    gui.output_list_emi.setPlainText(
        ", ".join(
            s.strip()
            for s in output.get("output_list_emi", "").split(",")
            if len(s.strip()) > 0
        )
    )
    gui.output_list_Vd.setPlainText(
        ", ".join(
            s.strip()
            for s in output.get("output_list_vd", "").split(",")
            if len(s.strip()) > 0
        )
    )
    gui.output_list_vap.setPlainText(
        ", ".join(
            s.strip()
            for s in output.get("output_list_vap", "").split(",")
            if len(s.strip()) > 0
        )
    )

    gui.description.setPlainText(guis.get("scenario_description", ""))


def _update_gui_from_compile_settings(settings, gui):
    guis = settings.get("nml_gui", dict())

    gui.sosaa_exe.setText(guis.get("sosaa_exe", "SOSAA.exe"))


def _update_gui_from_run_settings(settings, gui):
    guis = settings.get("nml_gui", dict())

    gui.launch_cmd.setText(
        guis.get("launch_cmd", "orterun --oversubscribe -n 4")
    )


def _update_gui_from_rsm_settings(settings, gui):
    rsm = settings.get("nml_rsm", dict())

    # Pretty-print the rsm path if relative to the working dir
    rsm_path = str(Path(rsm.get("rsm_path", "./sosaa-rsm.jl")).resolve())
    if rsm_path.startswith(str(Path.cwd())):
        rsm_path = f"./{Path(rsm_path).relative_to(Path.cwd())}"
    gui.rsm_path.setText(rsm_path)

    gui.rsm_train_seed.setText(rsm.get("train_seed", "my-train-seed"))
    gui.rsm_forest.setValue(rsm.get("forest_size", 16))
    gui.rsm_train_samples.setValue(rsm.get("train_samples", 1))

    # Pretty-print the rsm output path if relative to the working dir
    rsm_output = str(Path(rsm.get("rsm_output", "./rsm-output.jl")).resolve())
    if rsm_output.startswith(str(Path.cwd())):
        rsm_output = f"./{Path(rsm_output).relative_to(Path.cwd())}"
    gui.rsm_output.setText(rsm_output)

    gui.rsm_predict_seed.setText(rsm.get("predict_seed", "my-predict-seed"))
    gui.rsm_predict_samples.setValue(rsm.get("predict_samples", 1))

    gui.rsm_perturbation.setPlainText(
        rsm.get("predict_perturbation", "return inputs")
    )


def _update_gui_from_custom_settings(settings, gui):
    custom = settings.get("nml_custom", None)

    if custom is not None:
        i = 1

        for key, value in custom.items():
            getattr(gui, f"customKey_{i}").setText(str(key))
            getattr(gui, f"customVal_{i}").setText(str(value))

            i += 1

            if i > 30:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.setText("Excessive custom config")
                msg.setInformativeText(
                    "The INITFILE contains more than 30 properties in the "
                    + f"'NML_CUSTOM' namelist. Not all of the {len(custom)} "
                    + "properties are shown in the GUI."
                )
                msg.setWindowTitle("Warning loading INITFILE")
                msg.exec_()

                break

        for i in range(i, 31):
            getattr(gui, f"customKey_{i}").setText("")
            getattr(gui, f"customVal_{i}").setText("")
    else:
        for i in range(1, 31):
            getattr(gui, f"customKey_{i}").setText("")
            getattr(gui, f"customVal_{i}").setText("")


def _update_gui_from_raw_settings(gui, raw):
    gui.rawEdit.document().setPlainText(raw)
