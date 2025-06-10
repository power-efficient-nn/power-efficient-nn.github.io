import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full", css_file="./custom.css")


@app.cell(hide_code=True)
def _():
    import math
    import random
    from typing import Optional

    import marimo as mo
    import plotly.graph_objects as go

    # increase font size of plots
    import plotly.io as pio
    from plotly.graph_objects import Figure
    from plotly.subplots import make_subplots

    custom_template = pio.templates["plotly"]
    custom_template.layout.font.size = 16  # type: ignore[report]
    pio.templates["custom"] = custom_template
    pio.templates.default = "custom"

    mo.center(mo.image("public/llama.png", width=600))
    return Figure, Optional, go, make_subplots, math, mo, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Introduction of data types
    """,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Binary system

    On hardware, values are encoded as sequences of binary symbols: 0 and 1.
    """,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _text = mo.md(
        r"""
    ## Integer data types

    ### Definition:

    $$x=\sum_{j=0}^{I-1} i_{j} 2^{j}$$

    where $i_j$ are the bits encoding an integer with $I$ bits.

    ### Examples:
    """,
    )
    _img_uint = mo.image("public/data_type/uint.png", width=438)
    _img_fixedp = mo.image("public/data_type/fixedp.png", width=482)
    _img_int = mo.image("public/data_type/int.png", width=420)
    mo.vstack([_text, _img_uint, _img_fixedp, _img_int])
    return


@app.cell(hide_code=True)
def _(mo):
    def print_int(binary: str, *, signed: bool = False, num_frac_bits: int = 0):
        binary_orig = binary
        sign = 1
        if signed:
            sign_bit = binary[0]
            binary = binary[1:]
            if sign_bit == "1":
                sign = -1
        binary_rev = binary[::-1]
        pow_str = []
        value_str = []
        value = 0
        for index, digit in enumerate(binary_rev):
            if digit == "1":
                value_digit = 2 ** (index - num_frac_bits)
                pow_str.append(f"2^{{{index - num_frac_bits}}}")
                value_str.append(f"{value_digit}")
                value += value_digit
        value = sign * value
        value_ref = binary_to_int(binary_orig, signed=signed, num_frac_bits=num_frac_bits)
        if value != value_ref:
            msg = f"print ({value}) does not match with function to get value ({value_ref})"
            raise ValueError(msg)
        pow_str = "+".join(pow_str[::-1])
        value_str = "+".join(value_str[::-1])
        if pow_str == "":
            pow_str = "0"
            value_str = "0"
        if sign == -1:
            pow_str = f"-({pow_str})"
            value_str = f"-({value_str})"
        return mo.md(
            f"\\begin{{align}}"
            f"x &= \\text{{0b}}{binary} \\\\"
            f"&= {pow_str} \\\\"
            f"&= {value_str} \\\\"
            f"&= {value}"
            f"\\end{{align}}",
        )

    return (print_int,)


@app.function
def binary_to_int(binary: str, *, signed: bool = False, num_frac_bits: int = 0) -> float:
    sign = 1
    if signed:
        sign_bit = binary[0]
        binary = binary[1:]
        if sign_bit == "1":
            sign = -1
    binary = binary[::-1]
    value = 0
    for index, bit in enumerate(binary):
        if bit == "1":
            value += 2 ** (index - num_frac_bits)
    return sign * value


@app.cell(hide_code=True)
def _(mo):
    inttype_dropdown = mo.ui.dropdown(
        options={"UINT8", "FxP8", "INT8"},
        value="UINT8",
        label="Binary encoding of ",
    )
    binary_box = mo.ui.text(label=" value:", value="01111111")
    mo.vstack([inttype_dropdown, binary_box])
    return binary_box, inttype_dropdown


@app.cell(hide_code=True)
def _(binary_box, inttype_dropdown, print_int):
    signed = False
    num_frac_bits = 0
    int_type = inttype_dropdown.value
    if int_type == "INT8":
        signed = True
    elif int_type == "FxP8":
        num_frac_bits = 7
    binary = binary_box.value
    if len(binary) != 8:
        _msg = "provided binary must have 8 digits"
        raise ValueError(_msg)
    print_int(binary, signed=signed, num_frac_bits=num_frac_bits)
    return


@app.cell(hide_code=True)
def _(mo):
    _text = mo.md(
        r"""
    ## Floating-point data types

    Note that the introduced data types are simplified and not compliant with standards like IEEE.

    ### Definition:

    $$x=2^{e} (1 + \hat{m})=2^{e} (1 + 0.m)$$

    with $\hat{m} = \frac{m}{2^M}$.
    The exponent $e$ and mantissa $m$ values are integers encoded with $E$ and $M$ bits, respectively.

    ### Example:
    """,
    )
    _img = mo.image("public/data_type/fp.png", width=400)
    mo.vstack([_text, _img])
    return


@app.cell(hide_code=True)
def _(mo):
    def print_fp(sign_value: int, exp_value: int, mant_value: int, num_exp_bits: int, num_mant_bits: int):
        sign = -1 if sign_value else 1
        sign_str = "-" if sign_value else ""
        exponent_bias = -(2 ** (num_exp_bits - 1) - 1)
        mant_shift = 2**num_mant_bits
        exp_eff = exp_value + exponent_bias
        mant_eff = 1.0 + mant_value / mant_shift
        value = sign * (2**exp_eff) * mant_eff
        md = mo.md(
            f"\\begin{{align}}"
            f"x &= {{{sign_str}}} 2^{{{exp_value}{exponent_bias}}}(1 + \\frac{{{mant_value}}}{{{mant_shift}}}) \\\\"
            f"&= {{{sign_str}}} 2^{{{exp_eff}}} \\cdot {mant_eff} \\\\"
            f"&= {value}"
            f"\\end{{align}}",
        )
        value_ref = get_fp_value(sign_value, exp_value, mant_value, num_exp_bits, num_mant_bits)
        if value != value_ref:
            msg = f"print ({value}) does not match with function to get value ({value_ref})"
            raise ValueError(msg)
        return md

    return (print_fp,)


@app.function
def get_fp_value(sign_value: int, exp_value: int, mant_value: int, num_exp_bits: int, num_mant_bits: int) -> float:
    sign = -1 if sign_value else 1
    exponent_bias = -(2 ** (num_exp_bits - 1) - 1)
    mant_shift = 2**num_mant_bits
    return sign * (2 ** (exp_value + exponent_bias)) * (1.0 + mant_value / mant_shift)


@app.cell(hide_code=True)
def _(mo):
    fp_sign_box = mo.ui.text(label="Binary encoding of FP8 value:", value="0")
    fp_exp_box = mo.ui.text(value="1111")
    fp_mant_box = mo.ui.text(value="111")
    mo.hstack([fp_sign_box, fp_exp_box, fp_mant_box], justify="start")
    return fp_exp_box, fp_mant_box, fp_sign_box


@app.cell(hide_code=True)
def _(fp_exp_box, fp_mant_box, fp_sign_box, print_fp):
    fp_sign_binary = fp_sign_box.value
    fp_exp_binary = fp_exp_box.value
    fp_mant_binary = fp_mant_box.value
    if len(fp_sign_binary) != 1 or len(fp_exp_binary) != 4 or len(fp_mant_binary) != 3:
        _msg = "binary must have 1 sign, 4 exponent and 3 mantissa bits"
        raise ValueError(_msg)
    fp_sign = binary_to_int(fp_sign_binary)
    fp_exp = binary_to_int(fp_exp_binary)
    fp_mant = binary_to_int(fp_mant_binary)
    print_fp(fp_sign, fp_exp, fp_mant, 4, 3)
    return


@app.cell(hide_code=True)
def _(mo):
    _text = mo.md(
        r"""
    ## Logarithmic number system (LNS)

    ### Definition:

    $$x=2^{i + \hat{f}}=2^{i.f}$$

    with $\hat{f} = \frac{f}{2^F}$.
    The integer $i$ and fraction $f$ values are integers encoded with $I$ and $F$ bits, respectively.

    ### Example:
    """,
    )
    _img = mo.image("public/data_type/lns.png", width=400)
    mo.vstack([_text, _img])
    return


@app.cell(hide_code=True)
def _(mo):
    def print_lns(sign_value: int, int_value: int, frac_value: int, num_int_bits: int, num_frac_bits: int):
        sign = -1 if sign_value else 1
        sign_str = "-" if sign_value else ""
        exponent_bias = -(2 ** (num_int_bits - 1) - 1)
        frac_shift = 2**num_frac_bits
        exp = int_value + frac_value / frac_shift + exponent_bias
        value = sign * (2**exp)
        md = mo.md(
            f"\\begin{{align}}"
            f"x &= {{{sign_str}}} 2^{{{int_value} + \\frac{{{frac_value}}}{{{frac_shift}}}{exponent_bias}}} \\\\"
            f"&= {{{sign_str}}} 2^{{{exp}}} \\\\"
            f"&= {value}"
            f"\\end{{align}}",
        )
        value_ref = get_lns_value(sign_value, int_value, frac_value, num_int_bits, num_frac_bits)
        if value != value_ref:
            msg = f"print ({value}) does not match with function to get value ({value_ref})"
            raise ValueError(msg)
        return md

    return (print_lns,)


@app.function
def get_lns_value(sign_value: int, int_value: int, frac_value: int, num_int_bits: int, num_frac_bits: int) -> float:
    sign = -1 if sign_value else 1
    exponent_bias = -(2 ** (num_int_bits - 1) - 1)
    frac_shift = 2**num_frac_bits
    return sign * (2 ** (int_value + frac_value / frac_shift + exponent_bias))


@app.cell(hide_code=True)
def _(mo):
    lns_sign_box = mo.ui.text(label="Binary encoding of LNS8 value:", value="0")
    lns_int_box = mo.ui.text(value="1111")
    lns_frac_box = mo.ui.text(value="111")
    mo.hstack([lns_sign_box, lns_int_box, lns_frac_box], justify="start")
    return lns_frac_box, lns_int_box, lns_sign_box


@app.cell(hide_code=True)
def _(lns_frac_box, lns_int_box, lns_sign_box, print_lns):
    lns_sign_binary = lns_sign_box.value
    lns_int_binary = lns_int_box.value
    lns_frac_binary = lns_frac_box.value
    if len(lns_sign_binary) != 1 or len(lns_int_binary) != 4 or len(lns_frac_binary) != 3:
        _msg = "binary must have 1 sign, 4 log integer and 3 log fraction bits"
        raise ValueError(_msg)
    lns_sign = binary_to_int(lns_sign_binary)
    lns_int = binary_to_int(lns_int_binary)
    lns_frac = binary_to_int(lns_frac_binary)
    print_lns(lns_sign, lns_int, lns_frac, 4, 3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Visualization of quantization levels


    For simplicity, we study only positive values of signed data types.
    To this end, the first bit is hard-coded to zero in the following.
    """,
    )
    return


@app.cell(hide_code=True)
def _(Figure, go):
    def get_int_values(num_bits: int) -> list[float]:
        return [float(x) for x in range(2**num_bits)]

    def get_fp_values(num_exp_bits: int, num_mant_bits: int) -> list[float]:
        exp_values = range(2**num_exp_bits)
        mant_values = range(2**num_mant_bits)
        return [
            get_fp_value(0, exp_value, mant_value, num_exp_bits, num_mant_bits)
            for exp_value in exp_values
            for mant_value in mant_values
        ]

    def get_lns_values(num_int_bits: int, num_frac_bits: int) -> list[float]:
        int_values = range(2**num_int_bits)
        frac_values = range(2**num_frac_bits)
        return [
            get_lns_value(0, int_value, frac_value, num_int_bits, num_frac_bits)
            for int_value in int_values
            for frac_value in frac_values
        ]

    def normalize(values: list[float]) -> tuple[list[float], float]:
        values_max = max(values)
        values_normalized = [x / values_max for x in values]
        return (values_normalized, values_max)

    def plot_quant_levels(fig: Figure, index: int, values: list[float], name: str):
        fig.add_trace(go.Scatter(x=values, y=[index] * len(values), name=name, mode="markers"))

    def update_layout_qlevels(fig: Figure, normalized: bool):
        title = "Distribution of positive quantization levels"
        if normalized:
            title += " normalized to [0, 1]"
        fig.update_layout(
            title={"text": title},
            xaxis={"title": {"text": "value"}},
            yaxis={"visible": False},
            height=275,
        )

    return (
        get_fp_values,
        get_int_values,
        get_lns_values,
        normalize,
        plot_quant_levels,
        update_layout_qlevels,
    )


@app.cell(hide_code=True)
def _(mo):
    qlevels_normalized_dropdown = mo.ui.dropdown(
        options={"True": True, "False": False},
        value="False",
        label="Normalized:",
    )
    qlevels_normalized_dropdown
    return (qlevels_normalized_dropdown,)


@app.cell
def _(
    get_fp_values,
    get_int_values,
    get_lns_values,
    go,
    mo,
    normalize,
    plot_quant_levels,
    qlevels_normalized_dropdown,
    update_layout_qlevels,
):
    quant_levels_dict = {
        "LNS8": get_lns_values(4, 3),
        "FP8": get_fp_values(4, 3),
        "INT8": get_int_values(7),
    }

    _normalize = qlevels_normalized_dropdown.value
    _fig = go.Figure()
    for index, (name, quant_levels) in enumerate(quant_levels_dict.items()):
        qlevels = normalize(quant_levels)[0] if _normalize else quant_levels
        plot_quant_levels(_fig, index, qlevels, name)
    update_layout_qlevels(_fig, _normalize)
    mo.ui.plotly(_fig)
    return (quant_levels_dict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Visualization of quantization errors

    Note that quantizing a value by selecting the nearest quantization level
    is too computationally expensive in practice.
    Instead typically rounding is used.
    """,
    )
    return


@app.cell
def _():
    def get_closest_quant_value(value: float, quant_levels: list[float]) -> float:
        return min(quant_levels, key=lambda x: abs(x - value))

    def quantize(values: list[float], quant_levels: list[float]) -> list[float]:
        return [get_closest_quant_value(x, quant_levels) for x in values]

    return (quantize,)


@app.cell(hide_code=True)
def _(Figure):
    def get_error(value_true: float, value_pred: float, error_type: str) -> float:
        if error_type == "relative":
            error = abs((value_pred - value_true) / value_true) * 100.0
        elif error_type == "absolute":
            error = abs(value_pred - value_true)
        else:
            raise ValueError
        return error

    def get_errors(values_true: list[float], values_pred: list[float], error_type: str = "relative") -> list[float]:
        errors = []
        for value_true, value_pred in zip(values_true, values_pred):
            error = get_error(value_true, value_pred, error_type)
            errors.append(error)
        return errors

    def clamp_values_range(values: list[float], quant_levels: list[float]) -> list[float]:
        return [x for x in values if x >= min(quant_levels) and x <= max(quant_levels)]

    def get_error_max_str(name: str, error_max: float, error_type: str) -> str:
        error_max_str = f"{name}<br>max error: {error_max:.2f}"
        error_max_str = error_max_str + "%" if error_type == "relative" else error_max_str
        error_max_str += "<br>"
        return error_max_str

    def update_layout_qerror(fig: Figure, xaxis_type: str, normalized: bool, error_type: str):
        xaxis_label = "value"
        if normalized:
            xaxis_label = "normalized " + xaxis_label
        yaxis_label = error_type + " error"
        yaxis_label = yaxis_label + " (%)" if error_type == "relative" else yaxis_label
        fig.update_layout(
            title={"text": "Dynamic range and precision for different data types"},
            xaxis={"title": {"text": xaxis_label}, "type": xaxis_type},
            yaxis={"title": {"text": yaxis_label}},
        )

    return (
        clamp_values_range,
        get_error,
        get_error_max_str,
        get_errors,
        update_layout_qerror,
    )


@app.cell
def _(
    Figure,
    clamp_values_range,
    get_error_max_str,
    get_errors,
    go,
    normalize,
    quant_levels_dict,
    quantize,
    update_layout_qerror,
):
    def plot_quant_error(
        fig: Figure,
        values: list[float],
        quant_levels: list[float],
        error_type: str,
        normalized: bool,
        name: str,
    ):
        qlevels = normalize(quant_levels)[0] if normalized else quant_levels
        # remove zero to avoid division by zero in calculation of relative error
        if 0 in qlevels:
            qlevels = [x for x in qlevels if x != 0]
        values_clamped = clamp_values_range(values, qlevels)
        values_quant = quantize(values_clamped, qlevels)
        errors = get_errors(values_clamped, values_quant, error_type)
        error_max = max(errors)
        fig.add_trace(go.Scatter(x=values_clamped, y=errors, name=get_error_max_str(name, error_max, error_type)))

    def plot_quant_errors(fig: Figure, error_type: str, normalized: bool, xaxis_type: str):
        values = [2 ** (x / 1000.0 - 7.0) for x in range(16001)]  # [2**(-7), 2**9]
        if normalized:
            values = [x / (2**9) for x in values]  # [2**(-16), 2**0]
        for data_type in ["INT8", "FP8", "LNS8"]:
            quant_levels = quant_levels_dict[data_type]
            plot_quant_error(fig, values, quant_levels, error_type, normalized, data_type)
        update_layout_qerror(fig, xaxis_type, normalized, error_type)

    return (plot_quant_errors,)


@app.cell(hide_code=True)
def _(mo):
    errortype_dropdown = mo.ui.dropdown(options=["relative", "absolute"], value="absolute", label="Error type:")
    normalized_dropdown = mo.ui.dropdown(options={"True": True, "False": False}, value="False", label="Normalized:")
    xaxistype_dropdown = mo.ui.dropdown(options=["linear", "log"], value="linear", label="X-axis type:")
    mo.hstack([errortype_dropdown, normalized_dropdown, xaxistype_dropdown], justify="start")
    return errortype_dropdown, normalized_dropdown, xaxistype_dropdown


@app.cell(hide_code=True)
def _(
    errortype_dropdown,
    go,
    mo,
    normalized_dropdown,
    plot_quant_errors,
    xaxistype_dropdown,
):
    error_type = errortype_dropdown.value
    normalized = normalized_dropdown.value
    xaxis_type = xaxistype_dropdown.value
    _fig = go.Figure()
    plot_quant_errors(_fig, error_type, normalized, xaxis_type)
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Handling of outliers""")
    return


@app.cell(hide_code=True)
def _(Figure, random):
    def get_data_distrib(size: int, *, with_outlier: bool = False, reset_seed: bool = True) -> list[float]:
        if reset_seed:
            random.seed(42)
        values = [abs(random.gauss()) for _ in range(size)]
        if with_outlier:
            values[0] = abs(random.gauss()) * 30.0
        return values

    def update_layout_distrib(fig: Figure):
        fig.update_layout(
            title={"text": "Data distribution"},
            xaxis={"title": {"text": "value"}},
            yaxis={"title": {"text": "count"}},
        )

    return get_data_distrib, update_layout_distrib


@app.cell
def _(get_data_distrib, go, mo, num_feat, update_layout_distrib):
    _fig = go.Figure()
    _values = get_data_distrib(num_feat, with_outlier=False)
    _fig.add_trace(go.Histogram(x=_values, nbinsx=100))
    update_layout_distrib(_fig)
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Tensor-wise quantization""")
    return


@app.cell(hide_code=True)
def _(Figure, Optional, get_errors, go):
    def plot_rel_error(
        fig: Figure,
        values: list[float],
        values_quant: list[float],
        color: str,
        *,
        row: int = 1,
        col: int = 2,
    ):
        errors = get_errors(values, values_quant, "relative")
        fig.add_trace(
            go.Scatter(
                x=values,
                y=errors,
                mode="markers",
                marker_color=color,
                showlegend=False,
                legendgroup=color + "_qvalue",
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="original values", row=row, col=col)
        fig.update_yaxes(title_text="relative error (%)", row=row, col=col)

    def plot_scatter(
        fig: Figure,
        values: list[float],
        values_quant: list[float],
        color: str,
        *,
        quant_levels: Optional[list[float]] = None,
        show_identity: bool = True,
        row: int = 1,
        col: int = 1,
    ):
        xlim = [-max(values + values_quant) * 0.02, max(values + values_quant) * 1.02]
        if show_identity:
            fig.add_trace(
                go.Scatter(x=xlim, y=xlim, mode="lines", marker_color="black", opacity=0.2, name="identity"),
                row=row,
                col=col,
            )
        if quant_levels is not None:
            for index, quant_level in enumerate(quant_levels):
                fig.add_trace(
                    go.Scatter(
                        x=xlim,
                        y=[quant_level, quant_level],
                        mode="lines",
                        marker_color=color,
                        opacity=0.2,
                        showlegend=not bool(index),
                        legendgroup=color + "_qlevel",
                        name="quantization level",
                    ),
                    row=row,
                    col=col,
                )
        fig.add_trace(
            go.Scatter(
                x=values,
                y=values_quant,
                mode="markers",
                marker_color=color,
                legendgroup=color + "_qvalue",
                name="quantized values",
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="original values", range=xlim, row=row, col=col)
        fig.update_yaxes(title_text="quantized values", range=xlim, row=row, col=col)

    def update_layout_tensorq(fig: Figure):
        fig.update_layout(title={"text": "Tensor-wise quantization"})

    return plot_rel_error, plot_scatter, update_layout_tensorq


@app.cell
def _(
    Figure,
    Optional,
    get_data_distrib,
    normalize,
    plot_rel_error,
    plot_scatter,
    quantize,
):
    num_feat = 128

    def quantize_scaled(values: list[float], quant_levels_norm: list[float]) -> tuple[list[float], list[float]]:
        values_norm, scaling_factor = normalize(values)
        values_norm_quant = quantize(values_norm, quant_levels_norm)
        values_reconstr = [scaling_factor * x for x in values_norm_quant]
        quant_levels = [x * scaling_factor for x in quant_levels_norm]
        return values_reconstr, quant_levels

    def plot_error_scatter(
        fig: Figure,
        values: list[float],
        values_quant: list[float],
        color: str,
        *,
        quant_levels: Optional[list[float]] = None,
        show_identity: bool = True,
    ):
        plot_rel_error(fig, values, values_quant, color)
        plot_scatter(fig, values, values_quant, color, quant_levels=quant_levels, show_identity=show_identity)

    def plot_outliers(fig: Figure, quant_levels_norm: list[float], with_outlier: bool):
        values = get_data_distrib(num_feat, with_outlier=with_outlier)
        values_quant, quant_levels = quantize_scaled(values, quant_levels_norm)
        plot_error_scatter(fig, values, values_quant, "red", quant_levels=quant_levels)

    return num_feat, plot_error_scatter, plot_outliers, quantize_scaled


@app.cell(hide_code=True)
def _(mo, normalize, quant_levels_dict):
    datatype_dropdown = mo.ui.dropdown(
        options={data_type: normalize(quant_levels_dict[data_type])[0] for data_type in ["INT8", "FP8", "LNS8"]},
        value="INT8",
        label="Data type:",
    )
    outlier_dropdown = mo.ui.dropdown(options={"True": True, "False": False}, value="False", label="With outliers:")
    mo.hstack([datatype_dropdown, outlier_dropdown], justify="start")
    return datatype_dropdown, outlier_dropdown


@app.cell(hide_code=True)
def _(
    datatype_dropdown,
    make_subplots,
    mo,
    outlier_dropdown,
    plot_outliers,
    update_layout_tensorq,
):
    _fig = make_subplots(rows=1, cols=2)
    plot_outliers(_fig, datatype_dropdown.value, outlier_dropdown.value)
    update_layout_tensorq(_fig)
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Block-wise quantization""")
    return


@app.cell(hide_code=True)
def _(Figure):
    def update_layout_blockq(fig: Figure, index_block: int):
        fig.update_layout(title={"text": f"Block-wise quantization (block index {index_block})"})

    def update_layout_tensorvsblock(fig: Figure):
        fig.update_layout(title={"text": "Comparison between tensor- and block-wise quantization"})

    return update_layout_blockq, update_layout_tensorvsblock


@app.cell
def _(
    Figure,
    block_size,
    get_data_distrib,
    normalize,
    num_feat,
    plot_error_scatter,
    quant_levels_dict,
    quantize_scaled,
    update_layout_blockq,
):
    def get_block(values: list[float], index_block: int) -> list[float]:
        return values[index_block * block_size : (index_block + 1) * block_size]

    def plot_block_quant(fig: Figure, index_block: int, *, with_qlevels: bool = True):
        values = get_data_distrib(num_feat, with_outlier=True)
        quant_levels_norm = normalize(quant_levels_dict["INT8"])[0]

        # extract and quantize block
        values_block = get_block(values, index_block)
        values_block_quant, quant_levels_block = quantize_scaled(values_block, quant_levels_norm)

        # quantize tensor, then extract block
        values_quant, quant_levels = quantize_scaled(values, quant_levels_norm)
        values_quant_block = get_block(values_quant, index_block)

        plot_error_scatter(
            fig,
            values_block,
            values_quant_block,
            "red",
            quant_levels=quant_levels if with_qlevels else None,
        )
        plot_error_scatter(
            fig,
            values_block,
            values_block_quant,
            "blue",
            quant_levels=quant_levels_block if with_qlevels else None,
            show_identity=False,
        )
        update_layout_blockq(fig, index_block)

    return get_block, plot_block_quant


@app.cell(hide_code=True)
def _(mo):
    blocksize_dropdown = mo.ui.dropdown(
        options=["4", "8", "16", "32"],
        value="32",
        label="Block size:",
    )
    blocksize_dropdown
    return (blocksize_dropdown,)


@app.cell(hide_code=True)
def _(blocksize_dropdown, mo, num_feat):
    block_size = int(blocksize_dropdown.value)
    num_blocks = int(num_feat / block_size)
    index_slider = mo.ui.slider(start=0, stop=num_blocks - 1, label="Block index:", value=0)
    index_slider
    return block_size, index_slider, num_blocks


@app.cell(hide_code=True)
def _(index_slider, make_subplots, mo, plot_block_quant):
    _fig = make_subplots(rows=1, cols=2)
    plot_block_quant(_fig, index_slider.value)
    mo.ui.plotly(_fig)
    return


@app.cell
def _(
    Figure,
    get_block,
    get_data_distrib,
    make_subplots,
    mo,
    normalize,
    num_blocks,
    num_feat,
    plot_error_scatter,
    quant_levels_dict,
    quantize_scaled,
    update_layout_tensorvsblock,
):
    def plot_tensor_vs_block(fig: Figure):
        values = get_data_distrib(num_feat, with_outlier=True)
        quant_levels_norm = normalize(quant_levels_dict["INT8"])[0]

        # block-wise quantization
        values_block_quant = []
        for index_block in range(num_blocks):
            values_block = get_block(values, index_block)
            values_block_quant += quantize_scaled(values_block, quant_levels_norm)[0]

        # tensor-wise quantization
        values_tensor_quant = quantize_scaled(values, quant_levels_norm)[0]

        plot_error_scatter(fig, values, values_tensor_quant, color="red")
        plot_error_scatter(fig, values, values_block_quant, color="blue", show_identity=False)

    _fig = make_subplots(rows=1, cols=2)
    plot_tensor_vs_block(_fig)
    update_layout_tensorvsblock(_fig)
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Multiplierless dot products""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Log math

    Multiplications in linear space are additions in logarithmic space:

    $$\begin{align}
    a \cdot b &= 2^{\log_2(a)} 2^{\log_2(b)} \\
    &= 2^{\log_2(a) + \log_2(b)}
    \end{align}$$
    """,
    )
    return


@app.cell
def _(get_data_distrib, math, num_feat):
    values_a = get_data_distrib(num_feat, reset_seed=False)
    values_b = get_data_distrib(num_feat, reset_seed=False)

    def dot_product(values_a: list[float], values_b: list[float]) -> float:
        dot_product = 0
        for index in range(len(values_a)):
            dot_product += values_a[index] * values_b[index]
        return dot_product

    def dot_product_log(values_a: list[float], values_b: list[float]) -> float:
        dot_product = 0
        for index in range(len(values_a)):
            values_a_log = math.log2(values_a[index])
            values_b_log = math.log2(values_b[index])
            product_log = values_a_log + values_b_log
            dot_product += 2**product_log
        return dot_product

    result = dot_product(values_a, values_b)
    result_log = dot_product_log(values_a, values_b)

    print(f"Standard dot product: {result:.2f}")
    print(f"Dot product in log space: {result_log:.2f}")
    return dot_product, result, values_a, values_b


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Mitchell's approximation

    Using a first-order Taylor expansion the following hardware-friendly conversions can be derived:

    1. Conversion from linear to logarithmic space:

    $$\begin{align}
    i.f &= \log_2(x) \\
    &= \log_2(2^e (1 + 0.m)) \\
    &= e + \log_2(1 + 0.m) \\
    &\approx e + 0.m \\
    &= e.m
    \end{align}$$


    2. Conversion from logarithmic to linear space:

    $$\begin{align}
    2^e (1 + 0.m) &= 2^{i.f} \\
    &= 2^i 2^{0.f} \\
    &\approx 2^i (1 + 0.f)
    \end{align}$$
    """,
    )
    return


@app.cell(hide_code=True)
def _(Figure, go):
    def add_traces_mitchell(
        fig: Figure,
        values: list[float],
        values_lin2log_mitchell: list[float],
        values_log2lin_mitchell: list[float],
    ):
        xlim = [1.0, 2.0]
        for col in range(1, 3):
            fig.add_trace(
                go.Scatter(
                    x=xlim,
                    y=xlim,
                    mode="lines",
                    marker_color="black",
                    opacity=0.2,
                    name="identity",
                    showlegend=bool(col - 1),
                    legendgroup="identity",
                ),
                row=1,
                col=col,
            )
        fig.add_trace(go.Scatter(x=values, y=values_lin2log_mitchell, name="lin2log"), row=1, col=1)
        fig.update_xaxes(title_text="value in linear space", row=1, col=1)
        fig.update_yaxes(title_text="value in log space", row=1, col=1)
        fig.add_trace(go.Scatter(x=values, y=values_log2lin_mitchell, name="log2lin"), row=1, col=2)
        fig.update_xaxes(title_text="value in log space", row=1, col=2)
        fig.update_yaxes(title_text="value in linear space", row=1, col=2)

    def update_layout_mitchell(fig: Figure):
        fig.update_layout(title={"text": "Mitchell's approximation for e=1"})

    return add_traces_mitchell, update_layout_mitchell


@app.cell
def _(Figure, add_traces_mitchell, math):
    def lin2log_mitchell(value: float) -> float:
        mantissa, exponent = math.frexp(value)
        mantissa = 2 * mantissa - 1
        exponent = exponent - 1
        return exponent + mantissa

    def log2lin_mitchell(value: float) -> float:
        integer = math.floor(value)
        fraction = value - integer
        return 2**integer * (1.0 + fraction)

    def plot_mitchell(fig: Figure):
        values = [1.0 + x / 1000.0 for x in range(1, 1001, 1)]  # [1.0, 2.0]
        values_lin2log_mitchell = [2 ** lin2log_mitchell(x) for x in values]
        values_log2lin_mitchell = [math.log2(log2lin_mitchell(x)) for x in values]
        add_traces_mitchell(fig, values, values_lin2log_mitchell, values_log2lin_mitchell)

    return lin2log_mitchell, log2lin_mitchell, plot_mitchell


@app.cell(hide_code=True)
def _(make_subplots, mo, plot_mitchell, update_layout_mitchell):
    _fig = make_subplots(rows=1, cols=2)
    plot_mitchell(_fig)
    update_layout_mitchell(_fig)
    mo.ui.plotly(_fig)
    return


@app.cell
def _(
    get_error,
    lin2log_mitchell,
    log2lin_mitchell,
    result,
    values_a,
    values_b,
):
    def dot_product_mitchell(values_a: list[float], values_b: list[float]) -> float:
        dot_product = 0
        for index in range(len(values_a)):
            values_a_log = lin2log_mitchell(values_a[index])
            values_b_log = lin2log_mitchell(values_b[index])
            product_log = values_a_log + values_b_log
            dot_product += log2lin_mitchell(product_log)
        return dot_product

    result_mitchell = dot_product_mitchell(values_a, values_b)

    print(f"Dot product in log space with Mitchell approximation: {result_mitchell:.2f}")
    print(f"Relative error of Mitchell approximation (%): {get_error(result, result_mitchell, 'relative'):.2f}")
    return (dot_product_mitchell,)


@app.cell(hide_code=True)
def _(Figure, go):
    def plot_qerror(fig: Figure, error: float):
        xlim = [0, 128]
        fig.add_trace(go.Scatter(x=xlim, y=xlim, mode="lines", marker_color="black", opacity=0.2, name="identity"))
        fig.add_trace(
            go.Scatter(
                x=xlim,
                y=[0, xlim[1] * (1.0 - error)],
                mode="lines",
                marker_color="blue",
                opacity=0.2,
                legendgroup="qe",
                name="quantization error",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=xlim,
                y=[0, xlim[1] * (1.0 + error)],
                mode="lines",
                marker_color="blue",
                opacity=0.2,
                showlegend=False,
                legendgroup="qe",
            ),
        )
        fig.update_xaxes(title_text="standard dot product")
        fig.update_yaxes(title_text="dot product in log space using Mitchell")

    def plot_point(fig: Figure, result: float, result_mitchell: float, showlegend: bool):
        fig.add_trace(
            go.Scatter(
                x=[result],
                y=[result_mitchell],
                mode="markers",
                marker_color="red",
                showlegend=showlegend,
                legendgroup="dp",
                name="dot product",
            ),
        )

    def update_layout_mitchellerror(fig: Figure):
        fig.update_layout(title={"text": "Error of dot products using Mitchell's approximations"})

    return plot_point, plot_qerror, update_layout_mitchellerror


@app.cell
def _(
    Figure,
    dot_product,
    dot_product_mitchell,
    get_data_distrib,
    go,
    mo,
    num_feat,
    plot_point,
    plot_qerror,
    update_layout_mitchellerror,
):
    def add_point(fig: Figure, showlegend: bool):
        values_a = get_data_distrib(num_feat, reset_seed=False)
        values_b = get_data_distrib(num_feat, reset_seed=False)
        result = dot_product(values_a, values_b)
        result_mitchell = dot_product_mitchell(values_a, values_b)
        plot_point(fig, result, result_mitchell, showlegend)

    _fig = go.Figure()
    for index_sample in range(100):
        add_point(_fig, not bool(index_sample))
    lns8_qerror = 1 / 2 ** (1 + 3)  # 1 + 3 = half of 3rd fraction bit
    plot_qerror(_fig, lns8_qerror)
    update_layout_mitchellerror(_fig)
    mo.ui.plotly(_fig)
    return


if __name__ == "__main__":
    app.run()
