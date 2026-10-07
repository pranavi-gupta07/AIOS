obj-m := aios.o
KDIR ?= /lib/modules/$(shell uname -r)/build
CC_USER ?= cc

all: module aiosstat

module:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

aiosstat: aiosstat.c aios_uapi.h
	$(CC_USER) -O2 -Wall -Wextra -o $@ aiosstat.c

load: module
	sudo insmod aios.ko

unload:
	sudo rmmod aios

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
	rm -f aiosstat

.PHONY: all module load unload clean
