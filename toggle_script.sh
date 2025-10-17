#!/bin/bash
/usr/bin/dbus-send --session --dest=com.parakeet.Transcribe --type=method_call /com/parakeet/Transcribe com.parakeet.Transcribe.Toggle
