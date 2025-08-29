#import "../../defs.typ": *

#show: post.with(
  date: "2025-02-26",
  categories: (
    "admin",
  ),
)

// TODO: implement collapsible output
#let output(content) = {
  content
}

= Fedora: Downgrade kernel

Fedora's approach to kernel updates is like a 'rolling release'. Kernels are updated frequently, and this may break your system. After I applied Fedora updates, it installed the new kernel version `6.13.4`, causing the NVIDIA driver and CUDA drivers to #link("https://bugzilla.rpmfusion.org/show_bug.cgi?id=7187")[break].

Interestingly, Fedora's repo may not store previous kernel versions (this is when I considered switching to another Linux distro again), so the downgrading process is not straightforward. In this post, I want to summarize commands that worked for me.

#figure(
  image("cover.jpeg", width: 50%),
  caption: [Linux],
)

== Select another kernel in bootloader

Fedora stores multiple kernel versions. You can select another kernel in the GRUB menu during boot or specify the default kernel in GRUB configuration.

To list all installed kernels, run:

```bash
sudo grubby --info=ALL
```
#output[
  ```
  index=0
  kernel="/boot/vmlinuz-6.13.4-200.fc41.x86_64"
  args="ro rootflags=subvol=root rhgb quiet $tuned_params rd.driver.blacklist=nouveau modprobe.blacklist=nouveau"
  root="UUID=307f4fa2-e994-422a-b9a7-cb40b5a40331"
  initrd="/boot/initramfs-6.13.4-200.fc41.x86_64.img $tuned_initrd"
  title="Fedora Linux (6.13.4-200.fc41.x86_64) 41 (Workstation Edition)"
  id="4085d2970ec5418cb3db95ead4b10a3c-6.13.4-200.fc41.x86_64"
  index=1
  kernel="/boot/vmlinuz-6.12.15-200.fc41.x86_64"
  args="ro rootflags=subvol=root rhgb quiet $tuned_params rd.driver.blacklist=nouveau modprobe.blacklist=nouveau"
  root="UUID=307f4fa2-e994-422a-b9a7-cb40b5a40331"
  initrd="/boot/initramfs-6.12.15-200.fc41.x86_64.img $tuned_initrd"
  title="Fedora Linux (6.12.15-200.fc41.x86_64) 41 (Workstation Edition)"
  id="4085d2970ec5418cb3db95ead4b10a3c-6.12.15-200.fc41.x86_64"
  index=2
  kernel="/boot/vmlinuz-6.11.4-301.fc41.x86_64"
  args="ro rootflags=subvol=root rhgb quiet $tuned_params rd.driver.blacklist=nouveau modprobe.blacklist=nouveau"
  root="UUID=307f4fa2-e994-422a-b9a7-cb40b5a40331"
  initrd="/boot/initramfs-6.11.4-301.fc41.x86_64.img $tuned_initrd"
  title="Fedora Linux (6.11.4-301.fc41.x86_64) 41 (Workstation Edition)"
  id="4085d2970ec5418cb3db95ead4b10a3c-6.11.4-301.fc41.x86_64"
  index=3
  kernel="/boot/vmlinuz-0-rescue-4085d2970ec5418cb3db95ead4b10a3c"
  args="ro rootflags=subvol=root rhgb quiet rd.driver.blacklist=nouveau modprobe.blacklist=nouveau"
  root="UUID=307f4fa2-e994-422a-b9a7-cb40b5a40331"
  initrd="/boot/initramfs-0-rescue-4085d2970ec5418cb3db95ead4b10a3c.img"
  title="Fedora Linux (0-rescue-4085d2970ec5418cb3db95ead4b10a3c) 41 (Workstation Edition)"
  id="4085d2970ec5418cb3db95ead4b10a3c-0-rescue"
  ```
]

In my case, 3 kernel versions were available: `6.13.4` (`index=0`), `6.12.15` (`index=1`), and `6.11.4` (`index=2`).

Specify the default kernel with the corresponding index:

```bash
sudo grubby --set-default-index=1
```

To verify that the default kernel has been set correctly:

```bash
sudo grubby --default-index
```

After rebooting, verify if your system already has all necessary kernel packages installed. If everything works correctly, no further action may be needed.

However, if packages aren't available, you'll need to identify which kernel packages are required. Typically, `kernel-devel` is sufficient for compiling kernel modules.
Proceed to the next section to check available versions of `kernel-devel` and return here if necessary.

== Downgrade `kernel-devel`

After the reboot, check that the system is running the desired kernel version:

```bash
uname -r
```

To install kernel modules, you need to install the `kernel-devel` package that matches the kernel version.
As not all kernels may be available in Fedora repo, find available versions:

```bash
sudo dnf list kernel-devel
```
#output[
  ```
  Updating and loading repositories:
   Fedora 41 - x86_64 - Updates                                                                                                                                                                       100% |  24.3 KiB/s |  21.3 KiB |  00m01s
  Repositories loaded.
  Installed packages
  kernel-devel.x86_64 6.11.4-301.fc41 fedora
  kernel-devel.x86_64 6.13.4-200.fc41 updates
  ```
]

In my case, only `kernel-devel-6.11.4-301.fc41` was available. I needed to return to downgrade the kernel from `6.12.15` (`index=1`) to `6.11.4` (`index=2`):

+ Return to the previous kernel section, select the previous kernel version, and reboot.
+ After the reboot, continue with the `kernel-devel` downgrade.

=== Option 1: Explicitly install the previous version

Now you can install the available version of the `kernel-devel` package:

```bash
sudo dnf install kernel-devel-$(uname -r)
```

=== Option 2: Implicitly install the previous version

Alternatively, you can use the `dnf downgrade` command (it may work, or not):

```bash
sudo dnf downgrade kernel-devel
```

For compiling kernel modules, the `kernel-devel` package is typically sufficient.

== Downgrade `kernel-headers` (optional)

The `kernel-headers` package is generally used for compiling user-space applications and may not be necessary for kernel module compilation.
Not all headers might be available and sometimes it's even not possible to find matching `kernel-devel` and `kernel-headers`.

The process is similar to `kernel-devel`:

```bash
sudo dnf list kernel-headers
```

```bash
sudo dnf install kernel-headers-$(uname -r)
```

```bash
sudo dnf downgrade kernel-headers
```

== Pin kernel version

As Fedora packages are updated frequently, you could simply avoid running the `dnf upgrade` command to prevent kernel updates for a few days. There's a good chance that developers will fix the issue, allowing you to update to the latest kernel version later.
However, this is not an ideal solution.

You can pin the kernel version by excluding specific packages from updates:

```bash
nano /etc/dnf/dnf.conf
```

Add the following line:

```txt
exclude=kernel*
```
N.B. Do not forget that you prevent kernel updates. Add a reminder to your to-dos in a couple of weeks to remove this line if you want to update the kernel.

== Bonus: Reinstall NVIDIA driver

This instruction should work generally but I needed to reinstall the NVIDIA driver after downgrading the kernel.

1. You may want to check which driver (`nouveau` or `nvidia`) is loaded:

  ```bash
  lsmod | grep nouveau
  ```

  ```bash
  lsmod | grep nvidia
  ```

2. Remove NVIDIA drivers if they were installed

  ```bash
  sudo dnf remove "*nvidia*"
  ```

3. Install NVIDIA drivers:

  ```bash
  sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda
  ```

  > Alternatively, if NVIDIA drivers were installed, you can use the `dnf reinstall` command:
  ```bash
  sudo dnf reinstall akmod-nvidia xorg-x11-drv-nvidia-cuda
  ```

4. Wait for the NVIDIA kernel module (`nvidia-kmod`) to compile. This waiting time (a few minutes) is important; if you reboot and force a rebuild, it may not load correctly.

5. You can initiate kernel modules compilation and update forcefully via

  ```bash
  sudo akmods --force
  ```

  If the module failed to compile, remove the driver, and try to switch to another kernel version.

7. Check that the NVIDIA driver is present, this will print its version:

  ```bash
  modinfo -F version nvidia
  ```
  ```
  565.77
  ```

After that, my GPU was detected and CUDA was working.

```bash
nvidia-smi
```
#output[
  ```
  Wed Feb 26 21:18:46 2025
  +-----------------------------------------------------------------------------------------+
  | NVIDIA-SMI 565.77                 Driver Version: 565.77         CUDA Version: 12.7     |
  |-----------------------------------------+------------------------+----------------------+
  | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
  |                                         |                        |               MIG M. |
  |=========================================+========================+======================|
  |   0  NVIDIA GeForce RTX 4090        Off |   00000000:01:00.0 Off |                  Off |
  |  0%   46C    P8             14W /  450W |     277MiB /  24564MiB |      0%      Default |
  |                                         |                        |                  N/A |
  +-----------------------------------------+------------------------+----------------------+

  +-----------------------------------------------------------------------------------------+
  | Processes:                                                                              |
  |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
  |        ID   ID                                                               Usage      |
  |=========================================================================================|
  |    0   N/A  N/A      1735      G   /usr/bin/gnome-shell                          210MiB |
  |    0   N/A  N/A      1791      G   /usr/bin/Xwayland                               8MiB |
  +-----------------------------------------------------------------------------------------+
  ```
]
