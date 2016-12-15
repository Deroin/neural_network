package de.dero.neural_network.serialization;

import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import de.dero.neural_network.Weight;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

public class WeightListTypeAdapter extends TypeAdapter<List<Weight>> {

    @Override
    public void write(JsonWriter out, List<Weight> value) throws IOException {
        out.beginArray()
           .value(value.size())
           .endArray();
    }

    @Override
    public List<Weight> read(JsonReader in) throws IOException {
        in.beginArray();
        int size = in.nextInt();
        List<Weight> ret = new LinkedList<>();
        for (int i = 0; i < size; i++) {
            ret.add(new Weight());
        }
        in.endArray();
        return ret;
    }
}
