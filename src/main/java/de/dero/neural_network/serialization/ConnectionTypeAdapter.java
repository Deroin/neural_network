package de.dero.neural_network.serialization;

import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import de.dero.neural_network.Connection;

import java.io.IOException;

public class ConnectionTypeAdapter extends TypeAdapter<Connection> {

    @Override
    public void write(JsonWriter out, Connection value) throws IOException {
        out.beginArray()
           .value(value.getNeuronIndex())
           .value(value.getWeightIndex())
           .endArray();
    }

    @Override
    public Connection read(JsonReader in) throws IOException {
        in.beginArray();
        Connection ret = new Connection(in.nextInt(), in.nextInt());
        in.endArray();
        return ret;
    }
}
