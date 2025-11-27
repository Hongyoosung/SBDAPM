# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Classes for connecting to Unreal Engine instances. Either by creating new ones or connecting to an already running one.
"""

import logging
import subprocess
from typing import List, Optional, Tuple
import grpc
import socket


class UnrealConnection:
    """
    Abstract Base Class for a gRPC based connection to Unreal Engine.

    Parameters
    ----------
    url : str
        The url to connect to

    port : int
        The port on that URL to connect to

    Attributes
    ----------
    url: str
        The URL to connect to
    port: int
        The port on the URL to connect to
    channel: grpc.Channel
        The channel connecting to Unreal Engine on the chosen address
    """
    def __init__(self, url:str, port: int):
        self.channel : Optional[grpc.Channel] = None
        self.url = url
        self.port = port

    def close(self) -> None:
        """
        Close the Unreal Connection. Method must be safe to call multiple times.
        """
        if hasattr(self,"channel") and self.channel:
            self.channel.close()
            self.channel = None
    
    def __del__(self) -> None:
        """
        Destructor for Unreal Connections. Close the connection.
        """
        self.close()

    def start(self) -> None:
        """
        Open the Connection to Unreal Engine.
        """
        self.channel = grpc.secure_channel(
            self.address, grpc.local_channel_credentials()
        ).__enter__()

    @property
    def address(self) -> str:
        """
        Returns the address of the connection

        Returns
        -------
        str
            The address of the connection
        """
        return self.url + ":" + str(self.port)

    def connect_stubs(self, *stubs : List["grpc.Stub"]) -> List["grpc.Stub"]:
        """
        Connects the gRPC stubs to the Unreal Engine channel

        Parameters
        ----------
        *stubs : List["grpc.Stub"]
            The gRPC stubs to connect to the Unreal Engine channel
        """

        assert (
            self.channel != None
        ), "Connection has not been started, please create your channel before connecting gRPC stubs"
        return [stub(self.channel) for stub in stubs]

    @property
    def is_active(self) -> bool:
        """
        Returns whether the connection is active or not

        Returns
        -------
        bool
            Whether the connection is active or not
        """
        return self.channel != None
    
    def __bool__(self) -> bool:
        """
        Returns whether the connection is active or not

        Returns
        -------
        bool
            True iff the connection is active
        """
        return self.is_active

    def get_open_port(self, url:str) -> Tuple[socket.socket, int]:
        """
        Get an open port on the given URL

        Parameters
        ----------
        url : str
            The URL to get an open port on

        Returns
        -------
        socket.socket
            A socket object bound to the open port
        int
            The open port
        """
        if socket.has_ipv6:
            tcp_socket = socket.socket(socket.AF_INET6)
        else:
            tcp_socket = socket.socket(socket.AF_INET)
        tcp_socket.bind((url, 0))
        port = tcp_socket.getsockname()[1]
        return tcp_socket, port


class UnrealEditorConnection(UnrealConnection):
    """
    A connection to a running Unreal Editor instance.

    Parameters
    ----------
    url : str
        The URL to connect to
    port : int
        The port on that URL to connect to

    Raises
    ------
    AssertionError
        If the port is not supplied
    """

    def __init__(self, url:str, port:int):
        super().__init__(url, port)
        assert (
            self.port is not None
        ), "Port must be supplied to open a connection to an existing Unreal Process"


class StandaloneUnrealConnection(UnrealConnection):
    """
    A standalone connection that launches an Unreal Engine instance when started.

    Parameters
    ----------
    url : str
        The URL to connect to
    ue_path : str
        The path to the Unreal Engine executable
    headless_mode : bool
        Whether to run in headless mode
    port : int, optional
        The port to connect to, if None, an open port will be found
    map : str, optional
        The map to load. Defaults to the default map in the Unreal Engine project
    display_logs : bool, default=True
        Whether to display logs
    set_fps : int, optional
        Use a fixed fps while running, if None, no fixed timestep is used
    disable_script : bool, default=False
        Whether to disable the autolaunch script setting in the Unreal Engine Schola Plugin
    
    Attributes
    ----------
    executable_path : str
        The path to the Unreal Engine executable
    headless_mode : bool
        Whether to run in headless mode
    display_logs : bool
        Whether to display logs
    set_fps : int
        Use a fixed fps while running, if None, no fixed timestep is used
    env_process : subprocess.Popen, optional
        The process running the Unreal Engine. None if the process is not running
    disable_script : bool
        Whether to disable the autolaunch script setting in the Unreal Engine Schola Plugin
    map : str
        The map to load.  Defaults to the default map in the Unreal Engine project
    tcp_socket : socket.socket, optional
        A socket object bound to the open port. None if the port is supplied, and we don't need to open a new port
    
    """
    def __init__(
        self,
        url: str,
        executable_path: str,
        headless_mode: bool,
        port: Optional[int] = None,
        map: str = None,
        display_logs: bool = True,
        set_fps: Optional[int] = None,
        disable_script: bool = False,
    ):
        if port is None:
            self.tcp_socket, port = self.get_open_port(url)
        else:
            self.tcp_socket = None
            port = port
        super().__init__(url, port)
        self.executable_path = executable_path
        self.headless_mode = headless_mode
        self.display_logs = display_logs
        self.set_fps = set_fps
        self.env_process = None
        self.disable_script = disable_script
        #Note any maps we want to use here need to be added to the build via Project Settings>Packaging>Advanced> List of Maps...
        #or on the command line with the -Map flag for UnrealAutomationTool
        self.map = map

    def make_args(self) -> List[str]:
        """
        Make the arguments supplied to the Unreal Engine Executable.

        Returns
        -------
        List[str]
            The arguments to be supplied to the Unreal Engine Executable
        """
        args = [self.executable_path, "-UNATTENDED"]
        if self.headless_mode:
            args += ["-nullRHI"]
        else:
            args += ["-WINDOWED"]

        if self.map:
            args += [self.map]
        if self.display_logs:
            args += ["-LOG"]
        if not self.set_fps is None:
            args += ["-BENCHMARK"]
            args += ["-FPS=" + str(self.set_fps)]
        args += ["-ScholaPort", str(self.port)]
        if self.disable_script:
            args += ["-ScholaDisableScript"]
        return args

    def start(self) -> None:
        """
        Start the Unreal Engine process.

        Raises
        ------
        Exception
            If the subprocess is already running
        """
        if self.env_process != None:
            raise Exception("Subprocess already running")
        super().start()
        args = self.make_args()
        self.env_process = subprocess.Popen(args)

    def close(self) -> None:
        """
        Close the connection to the Unreal Engine. Kills the Unreal Engine process if it is running.
        """
        super().close()

        if self.tcp_socket != None:
            self.tcp_socket.close()

        if self.env_process != None:
            logging.info("killing subprocess")
            self.env_process.kill()
            # subprocess.run("TASKKILL /F /PID {pid} /T".format(pid=self.env_process.pid))

    @property
    def is_active(self) -> bool:
        #channel is still active and the unreal engine is still running
        return super().is_active and (not self.env_process is None) and (self.env_process.poll() == None)
