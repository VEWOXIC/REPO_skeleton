from logging import getLogger

import numpy as np
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """This is a conceptual class representation of a simple BLE device
    (GATT Server). It is essentially an extended combination of the
    :class:`bluepy.btle.Peripheral` and :class:`bluepy.btle.ScanEntry` classes

    :param client: A handle to the :class:`simpleble.SimpleBleClient` client
        object that detected the device
    :type client: class:`simpleble.SimpleBleClient`
    :param addr: Device MAC address, defaults to None
    :type addr: str, optional
    :param addrType: Device address type - one of ADDR_TYPE_PUBLIC or
        ADDR_TYPE_RANDOM, defaults to ADDR_TYPE_PUBLIC
    :type addrType: str, optional
    :param iface: Bluetooth interface number (0 = /dev/hci0) used for the
        connection, defaults to 0
    :type iface: int, optional
    :param data: A list of tuples (adtype, description, value) containing the
        AD type code, human-readable description and value for all available
        advertising data items, defaults to None
    :type data: list, optional
    :param rssi: Received Signal Strength Indication for the last received
        broadcast from the device. This is an integer value measured in dB,
        where 0 dB is the maximum (theoretical) signal strength, and more
        negative numbers indicate a weaker signal, defaults to 0
    :type rssi: int, optional
    :param connectable: `True` if the device supports connections, and `False`
        otherwise (typically used for advertising ‘beacons’).,
        defaults to `False`
    :type connectable: bool, optional
    :param updateCount: Integer count of the number of advertising packets
        received from the device so far, defaults to 0
    :type updateCount: int, optional
    """

    def __init__(self, cfg):
        """Constructor method
        """
        nn.Module.__init__(self)
        self.cfg = cfg
        self.num_nodes = cfg["model"]["num_nodes"]
        self.feature_dim = cfg["model"]["feature_dim"]
        self.output_dim = cfg["model"]["output_dim"]

        self.input_window = cfg["data"]["lookback"]
        self.output_window = cfg["data"]["horizon"]
        self.device = cfg["model"]["device"]
        self._logger = getLogger()
        self._scaler = cfg["data"]["scalar"]

        self.encoder = nn.Sequential(
            nn.Linear(
                self.input_window *
                self.num_nodes *
                self.feature_dim,
                64),
            nn.ReLU(),
            nn.Linear(
                64,
                16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                16, 64), nn.ReLU(), nn.Linear(
                64, self.output_window * self.num_nodes * self.output_dim), )

    def forward(self, input, target, input_time, target_time):
        """Returns a list of :class:`bluepy.blte.Service` objects representing
        the services offered by the device. This will perform Bluetooth service
        discovery if this has not already been done; otherwise it will return a
        cached list of services immediately..

        :param uuids: A list of string service UUIDs to be discovered,
            defaults to None
        :type uuids: list, optional
        :return: A list of the discovered :class:`bluepy.blte.Service` objects,
            which match the provided ``uuids``
        :rtype: list On Python 3.x, this returns a dictionary view object,
            not a list
        """

        # [batch_size, input_window, num_nodes, feature_dim]
        input = input.cpu()
        input_time = input_time.cpu()
        input_time = np.expand_dims(input_time, axis=-1)
        input = np.expand_dims(input, axis=-1)
        input = [input]
        idx = np.arange(self.cfg["model"]["num_nodes"])
        idx = torch.tensor(idx).to(self.device)
        input.append(input_time)
        input = np.concatenate(input, axis=-1)
        trainx = torch.from_numpy(input).to(self.device)
        trainx = trainx.reshape(
            -1, self.input_window * self.num_nodes * self.feature_dim
        )
        x = trainx.float()
        # [batch_size, output_window * num_nodes * feature_dim]
        x = self.encoder(x)  # [batch_size, 16]
        x = self.decoder(x)
        # [batch_size, output_window * num_nodes * output_dim]
        outputs = x.reshape(-1, self.output_window,
                            self.num_nodes, self.output_dim)
        outputs = torch.squeeze(outputs)
        return outputs
