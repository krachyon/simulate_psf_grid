import functools
import pickle

import astropy.io.fits
import numpy as np
import appdirs
import contextlib
import pathlib
from typing import Union
import os
from photutils.centroids import centroid_quadratic
from image_registration.fft_tools import upsample_image
import scopesim
import tempfile
from scopesim_templates import stars
from itertools import chain
from tqdm.auto import tqdm
import anisocado
from typing import TypeVar
import multiprocessing as mp

T = TypeVar('T')
scopesim_lock = mp.Lock()

# values you may want to change
scopesim_working_dir = pathlib.Path(appdirs.user_cache_dir('scopesim_workspace'))
if not scopesim_working_dir.exists():
    scopesim_working_dir.mkdir()
input_file = pathlib.Path('PSFs.pkl')

wavelength = 2.15  # μm
recenter_psf = True
psf_name = 'our_custom_psf'  # for checking scopesim effects
star_mag = 20  # how bright the generated sources are
xshift, yshift = 0, 0  # what pixel to hit with the center of the source

# helpers

@contextlib.contextmanager
def work_in(path: Union[str, pathlib.Path]):
    """A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.
    LICENSE: MIT
    from: https://code.activestate.com/recipes/576620-changedirectory-context-manager/
    """
    prev_cwd = pathlib.Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def download() -> None:
    """
    get scopesim files if not present in current directory
    :return: No
    """
    with work_in(scopesim_working_dir):
        if not pathlib.Path('inst_pkgs/MICADO').exists():
            scopesim.download_package(["locations/Armazones",
                                       "telescopes/ELT",
                                       "instruments/MICADO"])

@np.vectorize
def center_of_index(length: int) -> float:
    """given an array with extent length, what index hits the center of the array?"""
    return (length-1)/2


def center_of_image(img: np.ndarray) -> tuple[float, float]:
    """in pixel coordinates, pixel center convention

    (snippet to verify the numpy convention)
    img=np.random.randint(1,10,(10,10))
    y,x = np.indices(img.shape)
    imshow(img)
    plot(x.flatten(),y.flatten(),'ro')
    """
    assert len(img.shape) == 2
    # xcenter, ycenter
    return tuple(center_of_index(img.shape)[::-1])


@functools.lru_cache
def get_dummy_hdus() -> astropy.io.fits.HDUList:
    # the (0,0) here encodes an off-axis shift. Depending on what we're doing,
    # might have to change this to be varriable
    return anisocado.misc.make_simcado_psf_file([(0, 0)], [wavelength], N=10)


def make_psf_effect(input_array) -> scopesim.effects.Effect:
    # The scopesim effect expects a few Fits headers to be present, generate a psf where we
    # can transplant our PSF image
    img_hdu = get_dummy_hdus()[2]

    # astrometry gets more exciting when the PSF images are not centered correctly. Like those from
    # anisocado. This fixes this if it's needed.
    if recenter_psf:
        actual_center = np.array(centroid_quadratic(input_array, fit_boxsize=5))
        expected_center = np.array(center_of_image(input_array))
        xshift, yshift = expected_center - actual_center
        resampled = upsample_image(input_array, xshift=xshift, yshift=yshift).real
        img_hdu.data = resampled
    else:
        img_hdu.data = input_array

    filename = tempfile.NamedTemporaryFile('w', suffix='.fits').name
    img_hdu.header["WAVE0"] = wavelength
    img_hdu.writeto(filename)

    return scopesim.effects.FieldConstantPSF(
        name=psf_name,  # add name here
        filename=filename,
        wavelength=wavelength,
        psf_side_length=input_array.shape[0],
        strehl_ratio=0.5,  # no idea if that's used anywhere relevant...
    )


def setup_optical_train() -> tuple[int, scopesim.OpticalTrain]:
    micado = scopesim.OpticalTrain('MICADO')

    # This sub-pixel mode distorts the PSF slightly. (flux is assigned, /then/ convolved )
    # https://github.com/krachyon/ScopeSim/tree/model_eval_mine is more accurate but hacky
    micado.cmds["!SIM.sub_pixel.flag"] = True

    # disable built-in PSFs
    micado['relay_psf'].include = False
    micado['micado_ncpas_psf'].include = False

    # TODO atmospheric dispersion might be messed up. (splits star into 4 vertically shifted sources)
    #  uncomment to Ignore both dispersion and correction
    # micado['armazones_atmo_dispersion'].include = False
    # micado['micado_adc_3D_shift'].include = False

    # Todo This way of looking up the index is pretty stupid.
    element_idx = [element.meta['name'] for element in micado.optics_manager.optical_elements].index('default_ro')

    return element_idx, micado


def get_source(x_mas, y_mas) -> scopesim.Source:
    spectral_types = ['A0V']
    return stars(filter_name='MICADO/filters/TC_filter_K-cont.dat',
                   amplitudes=[star_mag],
                   spec_types=spectral_types,
                   x=[x_mas], y=[y_mas])


def flatten(l: list[list[T]]) -> list[T]:
    return [entry for sublist in l for entry in sublist]


if __name__ == '__main__':
    with open(input_file, 'rb') as f:
        psfs, positions = pickle.load(f)
    psfs = flatten(psfs)
    positions = flatten(positions)

    output_path = pathlib.Path('generated_images')
    if not output_path.exists():
        output_path.mkdir()
    with work_in(scopesim_working_dir):
        download()
        # make star in the center of image. vary this to get different subpixel-shifts

        target = get_source(xshift, yshift)

    # workstep
    def generate_image(psf):
        with work_in(scopesim_working_dir):
            # we can't seem to remove the effect again. So just create everything from scratch
            # each loop m(
            psf_effect = make_psf_effect(psf)
            # TODO in case you get weird errors from scopesim, add a lock around setup_optical_train()
            # There's something weird how it re-uses a socket to connect to a server...
            # disatvantage: Now the extra processes don't actually help that much
            with scopesim_lock:
                element_idx, micado = setup_optical_train()
            micado.optics_manager.add_effect(psf_effect, ext=element_idx)
            micado.observe(target)
            img = micado.readout()[0][1].data
            return img

    with mp.Pool() as p:
        imgs = p.map(generate_image, tqdm(psfs))

    for position, img in zip(positions, imgs):
        astropy.io.fits.ImageHDU(img).writeto(output_path/f'simulated_{position[0]}_{position[1]}.fits', overwrite=True)
